{ pkgs ? import <nixpkgs> { } }:
with pkgs;
mkShell {
  buildInputs = [ grass-sass nodejs simple-http-server djlint ];
}
