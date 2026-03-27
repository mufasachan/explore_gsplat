{
  inputs = {
    # because Python 3.10
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixgl,
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
        config = {
          allowUnfree = true;
          useCuda = true;
        };
        overlays = [ nixgl.overlay ];
      };
      cudaPkgs = pkgs.cudaPackages_12_4;
      libs = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.uv
          pkgs.python310
          pkgs.nixgl.auto.nixGLDefault
          cudaPkgs.cudatoolkit
        ];
        shellHook = ''
          export LD_LIBRARY_PATH="${libs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          export CUDA_PATH=${pkgs.cudatoolkit}
          alias uv='nixGL uv'
          uv sync
        '';
      };
    };
}
