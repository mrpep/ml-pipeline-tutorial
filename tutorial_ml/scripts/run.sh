ginpipe configs/base/train_dnn.gin \
        configs/datasets/gtzan_genre.gin \
        configs/features/wav_classification.gin \
        configs/models/cnn.gin \
        --module_list configs/imports \
        --project_name tutorial \
        --experiment_name gtzan \
        --mods SEED=10