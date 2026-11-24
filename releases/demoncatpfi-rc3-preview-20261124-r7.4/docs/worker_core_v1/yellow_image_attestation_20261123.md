# 镜像字段：负责人最终决定方案B

方案A已撤销，不再要求14份r5回执提供容器关联、启动时间或未换镜像声明。image_digest保留原null，登记注明“未观测（4.9 非必需）”；其他必需字段仍检查。镜像标签/Id若主动提供，只进入备注，不进入身份匹配。

下列只读命令仅供Native构建身份核对使用，不是r5.1登记前提。Native产物仍须核对镜像与CANN，不能用取消r5镜像门禁来放行未知Native库。输出不包含容器名、环境变量或挂载路径。

```bash
python - "$EXISTING_R5_CONTAINER" <<'PY'
import subprocess,json,sys,hashlib,datetime
raw=subprocess.check_output(['docker','inspect','--type','container',sys.argv[1]])
c=json.loads(raw)[0]
image=json.loads(subprocess.check_output(['docker','image','inspect',c['Image']]))[0]
print(json.dumps(dict(image_id=c['Image'],repo_digests=image.get('RepoDigests',[]),
 container_started_at=c['State']['StartedAt'],inspected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 inspect_sha256=hashlib.sha256(raw).hexdigest()),ensure_ascii=False))
PY
```
