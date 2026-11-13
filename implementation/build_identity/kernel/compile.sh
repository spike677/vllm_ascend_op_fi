set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /work/venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCENDC_LOG_LEVEL=0
export ASCEND_CUSTOM_OPP_PATH=/audit/vendor_visible_a1/vendors/fault28_fia_graph_transformer
cd /audit/build/kernel_a1
mkdir output
python /usr/local/Ascend/cann-9.1.0/python/site-packages/opc_tool/opc.py /audit/vendor_visible_a1/vendors/fault28_fia_graph_transformer/op_impl/ai_core/tbe/fault28_fia_graph_transformer_impl/dynamic/fused_infer_attention_score.py --main_func=fused_infer_attention_score --input_param=/audit/build/kernel_a1/input_param.json --soc_version=Ascend910B4 --output=/audit/build/kernel_a1/output --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic --tiling_key=103000000000000000,109000000000000000,109000000000400000,119000000000400000 --op_debug_config=dump_cce
