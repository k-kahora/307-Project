{
  description = "A simple Python project with NumPy and other packages";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixgl.url = "github:guibou/nixGL";
    nixgl.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, nixgl }: {
    devShells = {
      x86_64-linux = let
        pkgs = nixpkgs.legacyPackages.x86_64-linux;
      in {

        # Devshell 2 with Pandas and Islp
        project1 = pkgs.mkShell {
          buildInputs = [
            pkgs.python3
            pkgs.texlive.combined.scheme-full
            pkgs.nodePackages.pyright
            pkgs.python3Packages.pandas
            pkgs.python3Packages.matplotlib
            pkgs.python3Packages.torch
            pkgs.python3Packages.scikit-learn
            pkgs.python3Packages.venvShellHook
          ];
	  venvDir = "./.venv37";
	  shellHook = ''
            export PYTHONSTARTUP=/home/malcolm/Documents/Homework/datamining/Homework/homework2/startup.py
            if [ ! -d "$venvDir" ]; then
              ${pkgs.python3}/bin/python -m venv $venvDir
            fi
            source $venvDir/bin/activate
            pip install ISLP
            unset SOURCE_DATE_EPOCH
          '';
	  };
      };
    };
  };
}
