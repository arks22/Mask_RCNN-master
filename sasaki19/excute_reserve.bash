python3 filament.py train

python3 filament.py evaluate --model=last --eval_type=bbox --year=2016 >> result/eval/eval_log.txt
python3 filament.py evaluate --model=last --eval_type=segm --year=2016 >> result/eval/eval_log.txt

python3 plot_loss.py
python3 inspect_model.py last all
