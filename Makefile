.PHONY: create_cpu create_gpu create_all export_cpu export_gpu check_active_kernels install_libs


# CREAZIONE DEGLI AMBIENTI
create_cpu:
	conda env create -f lwad_cpu_env.yml

create_gpu:
	conda env create -f lwad_gpu_env.yml
	conda env config vars set -n lwad_gpu_env LD_LIBRARY_PATH='$$CONDA_PREFIX/lib:$$LD_LIBRARY_PATH'

create_all: create_cpu create_gpu


# ATTIVAZIONE DEGLI AMBIENTI
cpu_activate:
	@echo "Esegui il comando direttamente nella tua shell attuale:"
	@echo "  conda activate lwad_cpu_env"

gpu_activate:
	@echo "Esegui il comando direttamente nella tua shell attuale:"
	@echo "  conda activate lwad_gpu_env"


# ESPORTAZIONE DELLE CONFIGURAZIONI
export_cpu:
	conda env export -n lwad_cpu_env > new_lwad_cpu_env.yml

export_gpu:
	conda env export -n lwad_gpu_env > new_lwad_gpu_env.yml


# CONTROLLO DEI KERNEL ATTIVI
check_active_kernels:
	ps -eo pid,user,%mem,rss,command --sort=-%mem | grep "[i]pykernel"


# INSTALLAZIONE LIBRERIE IN libs/
install_libs:
	pip install -e .


# RUN DEGLI ESPERIMENTI (make exp1/make exp2...)
#   make exp2                        default
#   make exp2 DATASET=UNSW_BW15      log e summary dedicati, per istanze paralelle
#   DATASET: UNSW_BW15 | CICIDS2017 | CTU13 | CSECICIDS2018

exp%:
	nohup env PYTHONUNBUFFERED=1 \
	$(shell conda info --envs | awk '$$1=="lwad_gpu_env" {print $$NF}')/bin/python \
	notebooks/experiment.py \
		--only s$*/ \
		$(if $(DATASET),--dataset $(DATASET),) \
		--summary-file summary_s$*$(if $(DATASET),_$(DATASET),).csv \
		--verbose 1 \
		--resume \
	> res_$*$(if $(DATASET),_$(DATASET),).log 2>&1 &

clean:
	rm -rf results/
	rm res*.log

gpu_usage:
	nvidia-smi -i 0 \
		--query-gpu=utilization.gpu \
		--format=csv,noheader,nounits -l 1 | \
	head -n 30 | \
	awk '{sum+=$$1; n++} END {if (n>0) printf "Utilizzo medio GPU 0: %.2f%%\n", sum/n}'