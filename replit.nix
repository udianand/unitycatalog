
{pkgs}: {
  deps = [
    pkgs.jupyter
    pkgs.python312Packages.jupyter-core
    pkgs.python312Packages.notebook
    pkgs.libxcrypt
  ];
}
