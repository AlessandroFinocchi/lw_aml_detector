.PHONY: create_cpu create_gpu create_all export_cpu export_gpu check_active_kernels install_libs


# CREAZIONE DEGLI AMBIENTI
create_cpu:
	conda env create -f lwad_cpu_env.yml

create_gpu:
	conda env create -f lwad_gpu_env.yml
	conda env config vars set -n lwad_gpu_env LD_LIBRARY_PATH='$$CONDA_PREFIX/lib:$$LD_LIBRARY_PATH'

create_all: create_cpu create_gpu


# ATTIVAZIONE DEGLI AMBIENTI
activate_cpu:
	@echo "Esegui il comando direttamente nella tua shell attuale:"
	@echo "  conda activate lwad_cpu_env"

activate_gpu:
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

# RESOURCE USAGE
WINDOW ?= 30 # Finestra temporale di campionamento (in secondi)

# 1. Utilizzo medio GPU (%) su finestra di campionamento
gpu_usage:
	nvidia-smi -i 0 \
		--query-gpu=utilization.gpu \
		--format=csv,noheader,nounits -l 1 | \
	head -n $(WINDOW) | \
	awk '{sum+=$$1; n++} END {if (n>0) printf "Utilizzo medio GPU 0: %.2f%%\n", sum/n}'

# 2. Utilizzo medio CPU (%) su finestra di campionamento
cpu_usage:
	@echo "Campionamento CPU per $(WINDOW)s..."
	@vmstat 1 $$(($(WINDOW) + 1)) | \
		tail -n +4 | \
		awk '{sum += (100 - $$15); n++} END {if (n > 0) printf "Utilizzo medio CPU (%ds): %.2f%%\n", n, sum/n}'

# 3. Utilizzo medio RAM (%) e footprint istantaneo
ram_usage:
	@echo "Campionamento RAM per $(WINDOW)s..."
	@free -m -s 1 -c $(WINDOW) | \
		awk '/^Mem:/ { \
			pct = (1 - $$7/$$2) * 100; \
			sum += pct; n++; \
			last_used = $$2 - $$7; last_tot = $$2; \
		} \
		END { \
			if (n > 0) printf "Utilizzo medio RAM (%ds): %.2f%% (Attuale: %d/%d MiB)\n", n, sum/n, last_used, last_tot; \
		}'

# 4. Spazio occupato da ogni sottocartella di 1° livello + contesto storage
disk_usage:
	@echo "Occupazione sottocartelle di primo livello in $(CURDIR):"
	@du -h --max-depth=1 . 2>/dev/null | awk '$$2 != "."' | sort -hr
	@echo "---"
	@du -sh . | awk '{printf "Totale cartella corrente: %s\n", $$1}'
	@df -h . | awk 'NR==2 {printf "Filesystem (%s): %s usati su %s (%s liberi - %s)\n", $$6, $$3, $$2, $$4, $$5}'