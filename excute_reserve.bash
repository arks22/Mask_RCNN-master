cd sasaki12/
pwd
python3 filament.py evaluate --model=last --eval_type=bbox --year=2016
python3 filament.py evaluate --model=last --eval_type=segm --year=2016

cd ..
cd sasaki13
pwd
python3 filament.py evaluate --model=last --eval_type=bbox --year=2016
python3 filament.py evaluate --model=last --eval_type=segm --year=2016

cd ..
cd sasaki15
python3 filament.py train
