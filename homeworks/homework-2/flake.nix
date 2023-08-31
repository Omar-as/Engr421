{
  description = "Simple OpenGL Project";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... } @ inputs:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          python310
          python310Packages.numpy
          python310Packages.pandas
          python310Packages.scipy
          python310Packages.matplotlib
        ];
      };
    });
}
