export GPUS="1," # change to your gpu id

for name in olek
#for name in 377 386 387 392 393 394 olek vlad
do
    python train_net.py --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS} silent True
    python run.py --type evaluate --cfg_file configs/inb/inb_${name}.yaml exp_name inb_${name} gpus ${GPUS}
done