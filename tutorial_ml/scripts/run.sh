#ginpipe configs/base/train_dnn.gin \
#        configs/datasets/gtzan_genre.gin \
#        configs/features/wav_classification.gin \
#        configs/models/cnn.gin \
#        --module_list configs/imports \
#        --project_name tutorial \
#        --experiment_name gtzan_cnn

#ginpipe configs/base/train_dnn.gin \
#        configs/datasets/gtzan_genre.gin \
#        configs/features/encodecmaemel256_classification.gin \
#        configs/models/encodecmae_mlp.gin \
#        --module_list configs/imports \
#        --project_name tutorial \
#        --experiment_name gtzan_encodecmae_mlp \
#        --mods MAX_LR=0.00001

ginpipe configs/base/train_dnn.gin \
        configs/datasets/gtzan_genre.gin \
        configs/features/encodecmaemel256_classification.gin \
        configs/models/encodecmae_mlp.gin \
        --module_list configs/imports \
        --project_name tutorial \
        --experiment_name gtzan_encodecmae_mlp-basemel256toec-mixture \
        --mods MAX_LR=0.00001 "ENCODECMAE_PATH='/mnt/shared/alpha/home/lpepino/encodecmaes/encodecmae-private/encodecmae/experiments/base_model_melspec256toec/upstream_model/pretrain_checkpoints/last.ckpt'"