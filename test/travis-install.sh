### If you need to do a fresh install, enable following
#rm -rf $HOME/miniconda
#rm -rf $HOME/.cache/pip

### Regular install, check for cached first.
if [ -d "$HOME/miniconda" ]; then
    export PATH="$HOME/miniconda/bin:$PATH"
    source activate myenv
else
    if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
        wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    fi
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda config --add channels omnia --add channels conda-forge
    conda create -y -n myenv python=$TRAVIS_PYTHON_VERSION
    conda install -y -n myenv numpy pymbar pytest pytest-cov codecov
    conda install -y -n myenv ambertools=17.0 -c http://ambermd.org/downloads/ambertools/conda/
    source activate myenv
    pip install git+https://github.com/slochower/pAPRika@4f204a1a1190386d652e38464e45582d8734b705
fi

python --version
conda list

