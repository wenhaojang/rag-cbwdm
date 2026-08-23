from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))

from src.diagnostics.method_failure import require_diagnostic_output
from src.diagnostics.signed_teacher_v1 import build_signed_training_groups
from src.formal_provenance import sha256_path
from src.io_utils import load_yaml
from src.run_manifest import atomic_write_json,git_state,sha256_file,stable_hash,utc_now
from src.selector_cross_encoder import CrossEncoderSelector,build_selector_input,cbwdm_multitask_loss


def _seed(value:int)->None:
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)


def main()->None:
    p=argparse.ArgumentParser(description="Train experimental signed_selector_v1")
    p.add_argument("--config",required=True);p.add_argument("--run-dir",required=True)
    p.add_argument("--teacher",required=True);p.add_argument("--posteriors",required=True);p.add_argument("--retrieval",required=True)
    p.add_argument("--output-dir",required=True);p.add_argument("--model-name",default="/root/models/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--epochs",type=int,default=3);p.add_argument("--lr",type=float,default=2e-5);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-length",type=int,default=512);p.add_argument("--max-train-groups",type=int,default=None)
    p.add_argument("--teacher-temperature",type=float,default=.1);p.add_argument("--b-plus",type=float,default=.01);p.add_argument("--b-minus",type=float,default=.001)
    p.add_argument("--beta",type=float,default=.25);p.add_argument("--gamma",type=float,default=1.0)
    p.add_argument("--neutral-sample-policy",choices=["negative","ignore"],default="negative")
    p.add_argument("--device",default="auto");p.add_argument("--seed",type=int,default=13);p.add_argument("--resume",action="store_true");args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:raise ValueError("epochs and batch-size must be >= 1")
    if args.max_train_groups is not None and args.max_train_groups<1:raise ValueError("max-train-groups must be >= 1")
    fixed={"epochs":3,"lr":2e-5,"batch_size":8,"beta":.25,"gamma":1.0,"teacher_temperature":.1,"seed":13}
    changed={name:(getattr(args,name),value) for name,value in fixed.items() if getattr(args,name)!=value}
    if changed:raise ValueError(f"signed_selector_v1 first-round hyperparameters are frozen: {changed}")
    config_path=Path(args.config).resolve();load_yaml(config_path);run=Path(args.run_dir).resolve();output=require_diagnostic_output(run,Path(args.output_dir).resolve())
    signed_root=(run/"artifacts/diagnostics/method_failure_audit/signed_teacher_v1").resolve()
    try:output.relative_to(signed_root/"training")
    except ValueError as exc:raise ValueError(f"training output must be below {signed_root/'training'}") from exc
    teacher=Path(args.teacher).resolve();posteriors=Path(args.posteriors).resolve();retrieval=Path(args.retrieval).resolve();model=Path(args.model_name).resolve()
    teacher_manifest=teacher.parent/"manifest.json"
    if not teacher_manifest.is_file():raise FileNotFoundError(f"signed teacher manifest missing: {teacher_manifest}")
    teacher_contract=json.loads(teacher_manifest.read_text(encoding="utf-8")).get("contract",{})
    teacher_params=teacher_contract.get("parameters",{})
    expected_teacher={"b_plus":args.b_plus,"b_minus":args.b_minus,"neutral_sample_policy":args.neutral_sample_policy}
    mismatched={key:(teacher_params.get(key),value) for key,value in expected_teacher.items() if teacher_params.get(key)!=value}
    if mismatched:raise ValueError(f"Training supervision thresholds differ from signed teacher manifest: {mismatched}")
    contract={"variant":"signed_selector_v1_training","experimental":True,"config_sha256":sha256_file(config_path),
        "inputs":{"teacher":{"path":str(teacher),"sha256":sha256_file(teacher)},"teacher_manifest":{"path":str(teacher_manifest),"sha256":sha256_file(teacher_manifest)},"posteriors":{"path":str(posteriors),"sha256":sha256_file(posteriors)},
                  "retrieval":{"path":str(retrieval),"sha256":sha256_file(retrieval)},"model":{"path":str(model),"sha256":sha256_path(model)}},
        "parameters":{"epochs":args.epochs,"lr":args.lr,"batch_size":args.batch_size,"max_length":args.max_length,
            "max_train_groups":args.max_train_groups,"teacher_temperature":args.teacher_temperature,"loss_type":"cbwdm_multitask",
            "b_plus":args.b_plus,"b_minus":args.b_minus,"beta":args.beta,"gamma":args.gamma,
            "neutral_sample_policy":args.neutral_sample_policy,"seed":args.seed,"device":args.device}}
    fingerprint=stable_hash(contract);manifest_path=output/"training_manifest.json";checkpoint=output/"checkpoint"
    outputs=[output/"training_config.json",output/"training_history.json"]
    if args.resume and manifest_path.is_file():
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status")=="completed" and manifest.get("fingerprint")==fingerprint and checkpoint.is_dir() and all(path.is_file() for path in outputs):
            if manifest.get("checkpoint_sha256")==sha256_path(checkpoint) and all(manifest["outputs"][p.name]["sha256"]==sha256_file(p) for p in outputs):
                print(f"[signed_selector_v1_train] reused=true output={output}");return
        raise ValueError("Cannot resume signed selector training: fingerprint/checksum mismatch")
    if manifest_path.exists() or checkpoint.exists() or any(path.exists() for path in outputs):raise FileExistsError("Training artifact exists; use matching --resume or a new smoke/full directory")
    groups=build_signed_training_groups(teacher_path=teacher,posteriors_path=posteriors,retrieval_path=retrieval,max_train_groups=args.max_train_groups)
    _seed(args.seed);selector=CrossEncoderSelector(model_name=str(model),max_length=args.max_length,device=args.device)
    optimizer=torch.optim.AdamW(selector.model.parameters(),lr=args.lr);history=[]
    for epoch in range(1,args.epochs+1):
        selector.model.train();ordered=list(groups);random.Random(args.seed+epoch).shuffle(ordered);optimizer.zero_grad()
        totals={"loss":0.0,"ce":0.0,"rank":0.0,"valid_rank":0,"skipped_rank":0,"groups":0}
        for idx,group in enumerate(ordered,start=1):
            texts=[build_selector_input(group.query,group.selected_docs,doc) for doc in group.candidate_docs]
            scores=selector.score_texts(texts,batch_size=len(texts),requires_grad=True)
            loss,details=cbwdm_multitask_loss(scores,group.effective_gains,b_plus=args.b_plus,b_minus=args.b_minus,
                gamma=args.gamma,beta=args.beta,neutral_sample_policy=args.neutral_sample_policy)
            (loss/args.batch_size).backward();totals["loss"]+=float(loss.detach().cpu());totals["ce"]+=float(details["ce_loss"].detach().cpu());totals["rank"]+=float(details["rank_loss"].detach().cpu())
            totals["valid_rank"]+=int(details["valid_ranking_group"]);totals["skipped_rank"]+=int(details["skipped_ranking_group"]);totals["groups"]+=1
            if idx%args.batch_size==0 or idx==len(ordered):optimizer.step();optimizer.zero_grad()
            if idx%100==0:print(f"[signed_selector_v1_train] epoch={epoch} group={idx}/{len(ordered)} loss={float(loss.detach().cpu()):.6f}")
        history.append({"epoch":epoch,"avg_total_loss":totals["loss"]/totals["groups"],"avg_ce_loss":totals["ce"]/totals["groups"],
            "avg_rank_loss":totals["rank"]/totals["groups"],"valid_ranking_groups":totals["valid_rank"],"skipped_ranking_groups":totals["skipped_rank"]})
    output.mkdir(parents=True,exist_ok=True)
    selector.save_checkpoint(checkpoint,extra_config={"variant":"signed_selector_v1","experimental":True,"loss_type":"cbwdm_multitask",
        "epochs":args.epochs,"lr":args.lr,"batch_size":args.batch_size,"teacher_temperature":args.teacher_temperature,
        "b_plus":args.b_plus,"b_minus":args.b_minus,"beta":args.beta,"gamma":args.gamma,
        "neutral_sample_policy":args.neutral_sample_policy,"num_groups":len(groups)})
    training_config={**contract["parameters"],"model_name":str(model),"num_groups":len(groups),"terminal_groups":sum(g.is_terminal_state for g in groups),
        "explicit_harmful_negative_count":sum(c=="explicit_harmful_negative" for g in groups for c in g.supervision_classes)}
    atomic_write_json(outputs[0],training_config);atomic_write_json(outputs[1],{"epochs":history})
    atomic_write_json(manifest_path,{"schema_version":"rag_cbwdm_signed_selector_training_manifest.v1","status":"completed","completed":True,
        "fingerprint":fingerprint,"contract":contract,"diagnostic_only":True,"experimental":True,"num_groups":len(groups),
        "checkpoint_path":str(checkpoint),"checkpoint_sha256":sha256_path(checkpoint),"outputs":{p.name:{"path":str(p),"sha256":sha256_file(p)} for p in outputs},
        "git":git_state(PROJECT_ROOT),"completed_at":utc_now()})
    print(f"[signed_selector_v1_train] groups={len(groups)} checkpoint={checkpoint}")


if __name__=="__main__":main()
