//! TLS 1.3 between the mrun servers of a pool (026 R5), with no certificate authority: every
//! Mac presents its own self-signed certificate ([`Identity`]), and each side checks the other's
//! against its pin. A client either expects one pinned certificate or (while joining) accepts any
//! and hands it to the join's code. The server asks every client for a certificate and gives
//! handlers the peer's ([`Peer`]); the pool routes refuse non-members before reading a body.

use super::identity::{Identity, mac_id};
use std::sync::Arc;
use tokio_rustls::rustls::client::danger::{
    HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier,
};
use tokio_rustls::rustls::crypto::{CryptoProvider, aws_lc_rs};
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use tokio_rustls::rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use tokio_rustls::rustls::{self, DigitallySignedStruct, DistinguishedName, SignatureScheme};

fn provider() -> Arc<CryptoProvider> {
    Arc::new(aws_lc_rs::default_provider())
}

/// Accepts a certificate when it is the pinned one (or any, while joining) and its handshake
/// signature verifies.
#[derive(Debug)]
struct Pinned {
    pin: Option<Vec<u8>>,
    provider: Arc<CryptoProvider>,
}

impl Pinned {
    fn check(&self, end_entity: &CertificateDer<'_>) -> Result<(), rustls::Error> {
        match &self.pin {
            Some(pin) if pin.as_slice() != end_entity.as_ref() => Err(rustls::Error::General(
                "the certificate is not the one pinned for this Mac".into(),
            )),
            _ => Ok(()),
        }
    }

    fn signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        self.check(end_entity)
            .map(|()| ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Err(rustls::Error::General("TLS 1.3 only".into()))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.schemes()
    }
}

impl ClientCertVerifier for Pinned {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, rustls::Error> {
        self.check(end_entity)
            .map(|()| ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Err(rustls::Error::General("TLS 1.3 only".into()))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.schemes()
    }
}

fn chain(identity: &Identity) -> (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) {
    (
        vec![CertificateDer::from(identity.certificate.clone())],
        PrivateKeyDer::Pkcs8(identity.key.clone().into()),
    )
}

/// The server side: asks every client for its certificate (any; the routes check membership).
pub fn server_config(identity: &Identity) -> Result<Arc<rustls::ServerConfig>, String> {
    let provider = provider();
    let (certs, key) = chain(identity);
    let config = rustls::ServerConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| e.to_string())?
        .with_client_cert_verifier(Arc::new(Pinned {
            pin: None,
            provider,
        }))
        .with_single_cert(certs, key)
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(config))
}

/// The client side towards a Mac with certificate `pin`, or any Mac (`None`: joining).
pub fn client_config(
    identity: &Identity,
    pin: Option<Vec<u8>>,
) -> Result<rustls::ClientConfig, String> {
    let provider = provider();
    let (certs, key) = chain(identity);
    rustls::ClientConfig::builder_with_provider(provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| e.to_string())?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(Pinned { pin, provider }))
        .with_client_auth_cert(certs, key)
        .map_err(|e| e.to_string())
}

/// An HTTPS client towards one Mac (or any, joining). `tls_info` lets the join read the
/// server's certificate from a response.
pub fn client(identity: &Identity, pin: Option<Vec<u8>>) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .use_preconfigured_tls(client_config(identity, pin)?)
        .tls_info(true)
        .no_proxy()
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())
}

/// The server certificate a response came over.
pub fn server_certificate(response: &reqwest::Response) -> Option<Vec<u8>> {
    response
        .extensions()
        .get::<reqwest::tls::TlsInfo>()
        .and_then(|info| info.peer_certificate().map(<[u8]>::to_vec))
}

/// The Mac at the other end of a pool connection.
#[derive(Clone, Debug)]
pub struct Peer {
    pub certificate: Vec<u8>,
    pub mac_id: String,
    pub addr: std::net::SocketAddr,
}

/// A TLS listener for `axum::serve`: completes each handshake (or drops it after 5 s) before
/// the connection reaches the router.
pub struct Listener {
    tcp: tokio::net::TcpListener,
    acceptor: tokio_rustls::TlsAcceptor,
}

impl std::fmt::Debug for Listener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pool::tls::Listener({:?})", self.tcp.local_addr().ok())
    }
}

impl Listener {
    pub fn new(tcp: tokio::net::TcpListener, config: Arc<rustls::ServerConfig>) -> Self {
        Self {
            tcp,
            acceptor: tokio_rustls::TlsAcceptor::from(config),
        }
    }
}

impl axum::serve::Listener for Listener {
    type Io = tokio_rustls::server::TlsStream<tokio::net::TcpStream>;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok((tcp, addr)) = self.tcp.accept().await else {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            };
            let handshake = self.acceptor.accept(tcp);
            if let Ok(Ok(io)) =
                tokio::time::timeout(std::time::Duration::from_secs(5), handshake).await
            {
                return (io, addr);
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.tcp.local_addr()
    }
}

impl axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, Listener>> for Peer {
    fn connect_info(stream: axum::serve::IncomingStream<'_, Listener>) -> Self {
        let certificate = stream
            .io()
            .get_ref()
            .1
            .peer_certificates()
            .and_then(|c| c.first())
            .map(|c| c.as_ref().to_vec())
            .unwrap_or_default();
        Self {
            mac_id: mac_id(&certificate),
            certificate,
            addr: *stream.remote_addr(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keychain::Vault;
    use axum::extract::ConnectInfo;

    fn identity(dir: &std::path::Path) -> Identity {
        Identity::load_or_create(&Vault::dir(dir)).unwrap()
    }

    #[tokio::test]
    async fn pins_decide_who_talks() {
        let dir = tempfile::tempdir().unwrap();
        let (server, client_id, stranger) = (
            identity(&dir.path().join("a")),
            identity(&dir.path().join("b")),
            identity(&dir.path().join("c")),
        );
        let tcp = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = tcp.local_addr().unwrap();
        let app = axum::Router::new().route(
            "/who",
            axum::routing::get(|ConnectInfo(peer): ConnectInfo<Peer>| async move { peer.mac_id }),
        );
        let listener = Listener::new(tcp, server_config(&server).unwrap());
        tokio::spawn(async move {
            axum::serve(listener, app.into_make_service_with_connect_info::<Peer>())
                .await
                .unwrap();
        });
        let url = format!("https://{addr}/who");
        // Pinned to the server: the server sees the client's id.
        let pinned = client(&client_id, Some(server.certificate.clone())).unwrap();
        let who = pinned.get(&url).send().await.unwrap();
        assert_eq!(who.text().await.unwrap(), client_id.mac_id());
        // Joining (no pin): the response carries the server's certificate.
        let joining = client(&client_id, None).unwrap();
        let response = joining.get(&url).send().await.unwrap();
        assert_eq!(
            server_certificate(&response),
            Some(server.certificate.clone())
        );
        // A client pinned to another Mac refuses this server before any request is sent.
        let wrong = client(&client_id, Some(stranger.certificate.clone())).unwrap();
        assert!(wrong.get(&url).send().await.is_err());
    }
}
