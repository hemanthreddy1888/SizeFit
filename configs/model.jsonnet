{
    "sfnet": {
        "embedding_dim": 10,

        "num_item_emb": 5850 ,
        "num_category_emb": 7,
        "num_cup_size_emb": 12,
        "num_user_emb": 105508,

        "num_user_numeric": 4,
        "num_item_numeric": 2,

        "user_pathway": [256, 128, 64],
        "item_pathway": [256, 128, 64],
        "combined_pathway": [256, 128, 64, 16],

        "activation": "relu",
        "dropout": 0.3,

        "num_targets": 3
    },
    "trainer": {
        "num_epochs": 20,
        "batch_size": 128,
        "optimizer": {
            "lr": 0.001,
            "type": "adam",
            "weight_decay": 0.0001
        }
    },
    "logging": {
        "save_model_path": "runs/",
        "run_name": "trial_",
        "tensorboard": true,
        "print_every": 10
    }
}
