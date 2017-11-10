
wget https://www.dropbox.com/s/suhl3hkbsnbkpbo/s2vt?dl=0 -O s2vt
python src/predict.py models/data-processor.pickle s2vt "${1}" "${2}"
