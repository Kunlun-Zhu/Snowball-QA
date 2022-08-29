RUN_PRETRAIN = 0

if [$(RUN_PRETRAIN) -eq 1]; then:
sh run_train_ae.sh
sh run_train_qg.sh
fi

sh run_predict_ae.sh
sh run_predict_qg.sh
sh run_filter.sh
