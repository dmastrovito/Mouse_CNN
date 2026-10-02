#mkdir /scratch/fast/ImageNet
#cp /allen/programs/braintv/workgroups/cortexmodels/michaelbu/ImageNet/ILSVRC2016_CLS-LOC.tar.gz /scratch/fast/ImageNet
#cd /scratch/fast/ImageNet
#gunzip ILSVRC2016_CLS-LOC.tar.gz
#tar -xvf ILSVRC2016_CLS-LOC.tar.gz
#cp /allen/programs/mindscope/workgroups/tiny-blue-dot/mousenet/Mouse_CNN/data/imagenet/valprep.sh .
#. valprep.sh
#srun -p braintv --gres gpu:v100:1 --mem-per-gpu 32gb -t 6:00:00 --pty bash 
#export PYTHONPATH=$PYTHONPATH:/allen/programs/mindscope/workgroups/tiny-blue-dot/mousenet/Mouse_CNN/cmouse/exps/imagenet/
#cd /allen/programs/mindscope/workgroups/tiny-blue-dot/mousenet/Mouse_CNN/cmouse/
#exps/imagenet/run.sh 
python main.py --seed 42  --mask 3 --fixmask=2021
