git clone http://git.veripool.org/git/Verilog-Perl  # Only first time
cd Verilog-Perl
git pull        # Make sure we're up-to-date
git tag         # See what versions exist (recent GITs only)
#git checkout master      # Use development branch (e.g. recent bug fix)
#git checkout stable      # Use most recent release
#git checkout v{version}  # Switch to specified release version

perl Makefile.PL
# Ignore warning about README, this file will be generated
make
make test
make install
