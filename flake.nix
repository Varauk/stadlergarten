{
  description = "Erdbeermet development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.erdbeermetUpstream = {
    url = "github:david-schaller/Erdbeermet/main";
    flake = false;
  };
  inputs.erdbeermet = {
    url = "path:./erdbeermet";
    flake = false;
  };

  outputs = inputs@{ self, flake-utils, nixpkgs, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let

        pkgs = import nixpkgs { inherit system; };

        erdbeermetPkg =
          { lib, python3, fetchFromGitHub, src, version }:
          python3.pkgs.buildPythonPackage {
            pname = "erdbeermet";
            inherit version src;

            propagatedBuildInputs = with python3.pkgs; [
              numpy
              scipy
              matplotlib
              tqdm
              mypy
            ];

            doCheck = false;
          };

        erdbeermet = pkgs.callPackage erdbeermetPkg {
          src = inputs.erdbeermet;
          version = "local";
        };

        erdbeermetUpstream = pkgs.callPackage erdbeermetPkg {
          src = inputs.erdbeermetUpstream;
          version = "upstream";
        };

      in {
        devShell = self.devShells.${system}.local;

        devShells = {
          # Enter a devshell with the local library exposed [default]
          local = pkgs.mkShell {
            packages = [ erdbeermet ];
            inputsFrom = [ erdbeermet ];
          };

          # Enter a devshell with the upstream library exposed
          upstream = pkgs.mkShell {
            packages = [ erdbeermetUpstream ];
            inputsFrom = [ erdbeermetUpstream ];
          };
        };
      });
}
