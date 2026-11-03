# Tests packaging/homebrew/mrun.rb.tmpl without Homebrew (008 SC-007, T036): the formula is
# rendered with placeholder URLs and evaluated against a stub of the formula DSL, once per
# stubbed macOS version, and must pick the 15.0 tarball below 26.2 and the NAX tarball from
# 26.2 on (compared as versions, so 26.10 is above 26.2).
# Run: ruby packaging/homebrew/formula_test.rb
require "minitest/autorun"

# A version compared segment by segment, as Homebrew's MacOSVersion does.
class StubVersion
  include Comparable

  def initialize(text)
    @parts = text.to_s.split(".").map(&:to_i)
  end

  def <=>(other)
    other = StubVersion.new(other) unless other.is_a?(StubVersion)
    a = @parts.dup
    b = other.parts.dup
    a << 0 while a.size < b.size
    b << 0 while b.size < a.size
    a <=> b
  end

  protected

  attr_reader :parts
end

module OS
  def self.mac?
    true
  end
end

module MacOS
  class << self
    attr_accessor :stubbed

    def full_version
      StubVersion.new(stubbed)
    end
  end
end

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
  attr_reader :outputs

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
    outputs.fetch(command.sub("#{bin}/", ""))
  end

  def assert_match(pattern, text)
    @assertions.assert_match(pattern, text)
  end
end

class FormulaTemplateTest < Minitest::Test
  TEMPLATE = File.read(File.join(__dir__, "mrun.rb.tmpl"))

  def formula_for(macos)
    MacOS.stubbed = macos
    text = TEMPLATE
           .gsub("{{VERSION}}", "0.1.0")
           .gsub("{{URL_MACOS26}}", "macos26.tar.gz")
           .gsub("{{SHA256_MACOS26}}", "sha26")
           .gsub("{{URL_MACOS15}}", "macos15.tar.gz")
           .gsub("{{SHA256_MACOS15}}", "sha15")
           .sub("class Mrun < Formula", "Class.new(Formula) do")
    eval(text, binding, "mrun.rb") # rubocop:disable Security/Eval
  end

  def test_macos_15_gets_the_15_0_build
    formula = formula_for("15.7")
    assert_equal "macos15.tar.gz", formula.chosen_url
    assert_equal "sha15", formula.chosen_sha
  end

  def test_26_1_gets_the_15_0_build
    assert_equal "macos15.tar.gz", formula_for("26.1").chosen_url
    assert_equal "macos15.tar.gz", formula_for("26.0").chosen_url
  end

  def test_26_2_and_later_get_the_nax_build
    formula = formula_for("26.2")
    assert_equal "macos26.tar.gz", formula.chosen_url
    assert_equal "sha26", formula.chosen_sha
    assert_equal "macos26.tar.gz", formula_for("26.10").chosen_url
    assert_equal "macos26.tar.gz", formula_for("27.0").chosen_url
  end

  NAX = "mrun 0.1.0 (d1447fa, 2026-10-17) mlx 0.32.4 mlx-c 0.7.0 macos-arm64 " \
        "numerics nax compiled, available, gpu applegpu_g17s\n"
  NO_NAX = "mrun 0.1.0 (d1447fa, 2026-10-17) mlx 0.32.4 mlx-c 0.7.0 macos-arm64 " \
           "numerics no nax, not available, gpu applegpu_g16s\n"

  def run_formula_test(macos, version_line)
    formula = formula_for(macos)
    outputs = { "mrun --version" => version_line, "mrun completions bash" => "_mrun() {}\n" }
    TestRun.new(outputs, self).instance_exec(&formula.test_block)
  end

  def test_caveats_name_the_build_and_the_reinstall_only_for_the_15_0_build
    nax = formula_for("26.2").new.caveats
    assert_match "This is the macOS 26.2 build", nax
    refute_match "brew reinstall", nax
    old = formula_for("26.1").new.caveats
    assert_match "This is the macOS 15 build", old
    assert_match "`brew reinstall mrun` installs the macOS 26.2 (NAX) build", old
  end

  def test_formula_test_checks_the_numerics_of_the_chosen_build
    run_formula_test("26.2", NAX)
    run_formula_test("15.7", NO_NAX)
    run_formula_test("26.1", NO_NAX)
    assert_raises(Minitest::Assertion) { run_formula_test("26.2", NO_NAX) }
    assert_raises(Minitest::Assertion) { run_formula_test("15.7", NAX) }
  end
end
