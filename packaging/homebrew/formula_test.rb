# Tests packaging/homebrew/mrun.rb.in without Homebrew: the formula is rendered by
# scripts/release/formula.sh from a stand-in SHA256SUMS and evaluated against a stub of the
# formula DSL: one archive for every supported macOS, its checksum, and the `test do` block.
# Run: ruby packaging/homebrew/formula_test.rb
require "minitest/autorun"
require "open3"
require "tmpdir"

# The parts of Homebrew's Formula DSL the template uses; `url` and `sha256` are recorded.
class Formula
  class << self
    attr_reader :chosen_url, :chosen_sha, :test_block

    def url(value)
      @chosen_url = value
    end

    def sha256(value)
      @chosen_sha = value
    end

    def desc(*) end
    def homepage(*) end
    def version(*) end
    def license(*) end
    def depends_on(*) end
    def on_macos
      yield
    end

    def test(&block)
      @test_block = block
    end
  end
end

# Runs the formula's `test do` block against canned `mrun` output.
class TestRun
  def initialize(outputs, assertions)
    @outputs = outputs
    @assertions = assertions
  end

  def bin
    "/stub/bin"
  end

  def version
    "0.1.0"
  end

  def shell_output(command)
    @outputs.fetch(command.sub("#{bin}/", ""))
  end

  def assert_match(pattern, text)
    @assertions.assert_match(pattern, text)
  end
end

class FormulaTemplateTest < Minitest::Test
  SCRIPT = File.expand_path("../../scripts/release/formula.sh", __dir__)
  URL = "https://github.com/hupe1980/mrun/releases/download/v0.1.0/mrun-0.1.0-macos-arm64.tar.gz"
  SHA = "a" * 64

  # formula.sh's output for 0.1.0 with `sums` as SHA256SUMS: [stdout, stderr, status].
  def render(sums)
    Dir.mktmpdir do |dir|
      path = File.join(dir, "SHA256SUMS")
      File.write(path, sums)
      Open3.capture3(SCRIPT, "0.1.0", path)
    end
  end

  def rendered
    @rendered ||= begin
      out, err, status = render("#{SHA}  mrun-0.1.0-macos-arm64.tar.gz\n")
      assert status.success?, err
      out
    end
  end

  def formula
    eval(rendered.sub("class Mrun < Formula", "Class.new(Formula) do"), binding, "mrun.rb") # rubocop:disable Security/Eval
  end

  def test_one_archive_with_its_checksum
    f = formula
    assert_equal URL, f.chosen_url
    assert_equal SHA, f.chosen_sha
    refute_match(/full_version|macos15|macos26/, rendered)
  end

  def test_the_rendered_formula_has_no_placeholder
    refute_match(/@[A-Za-z0-9_-]*@/, rendered)
    assert_match 'version "0.1.0"', rendered
  end

  def test_a_missing_or_malformed_checksum_fails
    _, err, status = render("#{SHA}  mrun-0.1.0-macos26-arm64.tar.gz\n")
    refute status.success?
    assert_match "no checksum for mrun-0.1.0-macos-arm64.tar.gz", err
    _, _, status = render("xyz  mrun-0.1.0-macos-arm64.tar.gz\n")
    refute status.success?
  end

  def test_the_formula_test_checks_the_engine
    outputs = {
      "mrun --version" => "mrun 0.1.0 (c96fd4b, 2026-11-05) nuvec 0.6.1 macos-arm64, gpu applegpu_g16s, kernels precompiled, nax off (not an M5-class GPU)\n",
      "mrun completions bash" => "_mrun() {}\n",
    }
    TestRun.new(outputs, self).instance_exec(&formula.test_block)
    outputs["mrun --version"] = "mrun 0.1.0 (c96fd4b, 2026-11-05) macos-arm64\n"
    assert_raises(Minitest::Assertion) { TestRun.new(outputs, self).instance_exec(&formula.test_block) }
  end
end
