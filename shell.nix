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

  yarpPython = pkgs.yarp.overrideAttrs (old: {
    cmakeFlags = (old.cmakeFlags or []) ++ [
      "-DYARP_COMPILE_BINDINGS=ON"
      "-DCREATE_PYTHON=ON"
      "-DYARP_COMPILE_DEVICE_PLUGINS=OFF"
      "-DYARP_COMPILE_CARRIER_PLUGINS=OFF"
      "-DYARP_COMPILE_PORTMONITOR_PLUGINS=OFF"
      "-DYARP_COMPILE_TESTS=OFF"
      "-DPython3_EXECUTABLE=${python}/bin/python3"
    ];
    nativeBuildInputs = (old.nativeBuildInputs or []) ++ [ pkgs.swig python ];
  });

in
pkgs.mkShell {
  packages = [
    py
    yarpPython
    pkgs.fish
    pkgs.which
    pkgs.tcl
    pkgs.tk
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.libGL
    pkgs.libGLU
    pkgs.mesa
    pkgs.xorg.libX11
    pkgs.xorg.libXext
    pkgs.xorg.libXrender
    pkgs.xorg.libXfixes
    pkgs.xorg.libXcursor
    pkgs.xorg.libXi
    pkgs.xorg.libXrandr
    pkgs.xorg.libXinerama
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib
      pkgs.libGL
      pkgs.libGLU
      pkgs.mesa
      pkgs.xorg.libX11
      pkgs.xorg.libXext
      pkgs.xorg.libXrender
      pkgs.xorg.libXfixes
      pkgs.xorg.libXcursor
      pkgs.xorg.libXi
      pkgs.xorg.libXrandr
      pkgs.xorg.libXinerama
    ]}:$LD_LIBRARY_PATH"

    export PYTHONPATH="$(find ${yarpPython} -type d -path '*/site-packages' | head -n1):$PYTHONPATH"

    python3 -c "import yarp; print('yarp ok, Network=', hasattr(yarp,'Network'))"
    python3 -c "import pybullet as p; print('pybullet ok')"
    exec fish
  '';
}
