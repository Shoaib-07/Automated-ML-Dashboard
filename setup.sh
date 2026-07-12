#!/bin/bash
apt-get update
apt-get install -y python3.10 python3.10-venv python3.10-dev
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
