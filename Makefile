.PHONY: create_cpu create_gpu create_all export_cpu export_gpu check_active_kernels

# 1. CREAZIONE DEGLI AMBIENTI
create_cpu:
	conda env create -f lwad_cpu_env.yml

create_gpu:
	conda env create -f lwad_gpu_env.yml
	conda env config vars set -n lwad_gpu_env LD_LIBRARY_PATH='$$CONDA_PREFIX/lib:$$LD_LIBRARY_PATH'

create_all: create_cpu create_gpu


# 2. ATTIVAZIONE DEGLI AMBIENTI
cpu_activate:
	@echo "Esegui il comando direttamente nella tua shell attuale:"
	@echo "  conda activate lwad_cpu_env"

gpu_activate:
	@echo "Esegui il comando direttamente nella tua shell attuale:"
	@echo "  conda activate lwad_gpu_env"


# 3. ESPORTAZIONE DELLE CONFIGURAZIONI
export_cpu:
	conda env export -n lwad_cpu_env > new_lwad_cpu_env.yml

export_gpu:
	conda env export -n lwad_gpu_env > new_lwad_gpu_env.yml


# 4. CONTROLLO DEI KERNEL ATTIVI
check_active_kernels:
	ps -eo pid,user,%mem,rss,command --sort=-%mem | grep "[i]pykernel"