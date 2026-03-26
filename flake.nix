{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
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
        config.allowUnfree = true;
        overlays = [ nixgl.overlay ];
      };
      # Needed by torch
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
        ];
        shellHook = ''
          export LD_LIBRARY_PATH="${libs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          alias uv='nixGL uv'
          uv sync
        '';
      };
    };
}
