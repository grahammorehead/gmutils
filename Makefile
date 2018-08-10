# Makefile for gmutils

#########################################################################################################################
# Data

serialize_and_save_conceptnet_vectorfile:
	python gmutils/utils.py --file $(HOME)/data/ConceptNet/numberbatch-17.06.txt --pklfile $(HOME)/data/ConceptNet/numberbatch_en.pkl


#########################################################################################################################
# Admin

com:
	git commit -a -m "minor updates"
	git push

clean:
	# Clean up some artifacts created by other software typically on a Mac
	find . -name Icon$$'\r' -exec rm -fr {} \;
	find . -name '*.pyc' -exec rm -fr {} \;
	find . -name '*~' -exec rm -fr {} \;
	find . -name "*\[Conflict\]*" -exec rm -r {} \;
	find . -name "* \(1\)*" -exec rm -r {} \;
	find . -name ".DS_Store" -exec rm -r {} \;
	find . -name '__pycache__' -exec rm -fr {} \;


force:


#########################################################################################################################
# Configure

install_mac1:
	# Suggested version of Python 3 as of this writing:
	brew install https://raw.githubusercontent.com/Homebrew/homebrew-core/ec545d45d4512ace3570782283df4ecda6bb0044/Formula/python3.rb
	wget https://bootstrap.pypa.io/get-pip.py
	pip install --upgrade pip
	# By hand: (Using Python 3)
	# python -m venv ~/envs/sample
	# Add following lines to .zshrc \
alias rm="rm -i" \
alias mv="mv -i" \
alias cp="cp -i" \
alias python=python3 \
alias python2=/usr/bin/python \
unsetopt share_history \
setopt no_share_history


install_mac2:
	# sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
	python -m spacy download 'en_core_web_lg'
	defaults write -g InitialKeyRepeat -int 15 # normal minimum is 15 (225 ms)
	defaults write -g KeyRepeat -int 2 # normal minimum is 2 (30 ms)


install_ubuntu1:
	# By hand:
	# sudo apt-get update
	# sudo apt-get install -y build-essential cython python-numpy
	# sudo apt-get install -y zsh
	# sudo chsh -s /usr/bin/zsh ubuntu
	# zsh
	# sudo sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"
	# sudo apt-get upgrade  (if prompted, select "Package maintainer's version")
	# (might need) add contents of /etc/hostname to localhost line of /etc/hosts
	# reboot
	mkdir envs
	mkdir downloads
	sudo apt-get install -y mosh
	sudo apt-get install -y python3-dev
	sudo apt-get install -y emacs
	sudo apt-get install -y python3-tk
	sudo apt-get install -y python3-venv


# Latest Elastic Search as of this writing:
ES_DIR = https://artifacts.elastic.co/downloads/elasticsearch
ES_FILE = elasticsearch-6.2.4.deb
ENV     = default
install_ubuntu2:
	sudo apt-get install -y default-jdk
	wget -P downloads $(ES_DIR)/$(ES_FILE)
	sudo dpkg -i downloads/$(ES_FILE)
	# next command merely enables, not starts
	sudo systemctl enable elasticsearch.service
	python3 -m venv ../envs/$(ENV)
	source ../envs/$(ENV)/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install awscli
	pip install nltk
	pip install pip-review
	pip-review --local --interactive
	python -m spacy download en_core_web_lg
	# if desired
	# sudo emacs /etc/elasticsearch/elasticsearch.yml
	# sudo systemctl start elasticsearch


#########################################################################################################################
#########################################################################################################################
