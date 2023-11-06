{ pkgs ? import <nixpkgs> {} }:

with pkgs;

let
  pythonPackages = python310Packages; # You can change the Python version here

in

stdenv.mkDerivation {
  name = "python3-dev-env";
  buildInputs = [
    git
    gcc
    stdenv.cc.cc.lib
    poetry
    pythonPackages.python
    pythonPackages.pip
    pythonPackages.virtualenv
    pythonPackages.attrs
    pythonPackages.networkx
    pythonPackages.funcy
    pythonPackages.bidict
    pythonPackages.python-sat
    pythonPackages.more-itertools

  ];
  LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
}
