{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python311;

  ikpy = python.pkgs.buildPythonPackage rec {
    pname = "ikpy";
    version = "3.4.2";
    format = "pyproject";

    src = pkgs.fetchPypi {
      inherit pname version;
      hash = "sha256-98ycVAVXNd6viaec5nbm9ayrxylU52nRBD/r/b+1eEA=";
    };

    nativeBuildInputs = with python.pkgs; [ setuptools wheel pip ];
    propagatedBuildInputs = with python.pkgs; [ numpy scipy sympy ];
    doCheck = false;
  };

  py = python.withPackages (ps: with ps; [
    numpy
    scipy
    sympy
    pyyaml
    pybullet
    matplotlib
    tkinter
  ] ++ [ ikpy ]);

in
pkgs.mkShell {
  packages = [
    py
    pkgs.yarp
    pkgs.fish
    pkgs.which
    pkgs.tcl
    pkgs.tk
  ];

  shellHook = ''
    python -c "import numpy, scipy, sympy, ikpy, yaml, pybullet, matplotlib; print('ok: wszystkie paczki')"
    exec fish
  '';
}
