# BIRD-Critic InfiniSynapse Runner (Flash)

将 BIRD-Critic **Flash（PostgreSQL，200 题）** 提交给 InfiniSynapse Agent 执行，
并把修复后的 SQL 回收为可评估的 `pred_sqls` jsonl。
参考 `Spider2/methods/spider_agent_infini` 的提交/轮询/回收流程裁剪而来。

## 前置条件

1. InfiniSynapse 上已注册与 Flash `db_id` 同名的 **PostgreSQL 数据源**
   （名称匹配按小写、`-`→`_` 规范化），共 12 个：
   `financial`, `card_games`, `european_football_2`, `superhero`, `formula_1`,
   `student_club`, `codebase_community`, `debit_card_specializing`,
   `toxicology`, `california_schools`, `thrombosis_prediction`, `erolp`。
   本 runner **不做**数据源注册。
2. GT 数据已生成：`../evaluation/data/flash.jsonl`
   （由 `evaluation/data/build_gt_dataset.py` 产出）。
3. 至少一个可用的 InfiniSQL engine。

## 配置凭证

```bash
cd infini
cp infini_credential.json.example infini_credential.json  # 填入真实 api_key
pip install -r requirements.txt
```

也可用环境变量覆盖：`INFINI_API_URL` / `INFINI_API_KEY` / `INFINI_CONSOLE_URL` /
`INFINI_CREDENTIAL_PATH`。

## 运行

```bash
python run_flash.py --instance_id 0        # 单题冒烟
python run_flash.py --range 1,10           # 第 1-10 行
python run_flash.py --db_id financial      # 只跑某个库
python run_flash.py                        # 全量（自动跳过已有 pred 的题）
python run_flash.py --rerun                # 强制重跑
python run_flash.py --engine my-engine     # 指定 engine（并发 = engine 数）
```

每题流程：解析 `db_id` 对应数据源 → `newTask`（`databaseIds` 定向）→
轮询完成 → 下载 workspace zip → 抽取 `{instance_id}.sql` →
写入 `output/pred/flash.jsonl`。

多语句交付物用 `-- [BIRD_SPLIT]` 行分隔（prompt 中已约定），
避免按分号切坏 `$$ ... $$` 函数体。

## 产出与评估

- 预测文件：`output/pred/flash.jsonl`，格式 `{"instance_id": ..., "pred_sqls": [...]}`
- 各题原始 workspace：`output/<instance_id>/workspace/`
- 日志：`logs/`

合并进 GT 后走 docker 评估：

```bash
python ../evaluation/data/merge_pred_into_gt.py \
  --base ../evaluation/data/flash.jsonl \
  --pred output/pred/flash.jsonl \
  --out ../evaluation/data/flash_pred.jsonl
```

然后在 `evaluation/run/run_eval.sh` 中把 jsonl 指到
`/app/data/flash_pred.jsonl`、`dialect="postgresql"`、`mode="pred"`。

## 目录

```
infini/
  run_flash.py                     # 主入口
  bird_agent_infini/
    prompt.py                      # BIRD 调试 prompt（含 [BIRD_SPLIT] 约定）
    harvest.py                     # workspace -> pred_sqls 回收
    api/
      client.py                    # InfiniClient（凭证/HTTP）
      database.py                  # 数据源查询 / newTask / wait / downloadZip
  infini_credential.json           # 真实凭证（gitignore）
  output/                          # 运行产物（gitignore）
  logs/                            # 日志（gitignore）
```
