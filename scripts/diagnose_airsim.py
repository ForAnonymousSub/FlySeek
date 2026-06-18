# SPDX-License-Identifier: MIT
# Copyright (c) 2026 JoshuaWen
#
# Part of FlySeek: an adversarial aerial visual-language tracking (VLT)
# benchmark, built as a non-intrusive extension of OpenFly-Platform.
"""Diagnose AirSim API availability against the running AirVLN binary.

Why this script exists: AirVLN binaries (compiled ~2023) sometimes silently
return empty lists from `simListAssets()` because the underlying AirSim
plugin version is older than the API. Without this diagnostic you can't
distinguish "pak didn't load" from "List API doesn't work".

This script probes APIs one by one and prints which actually function,
so we can decide the right strategy for spawn/capture.

Usage:
    python scripts/diagnose_airsim.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _try(label: str, fn):
    """Run fn(), print compact OK/FAIL + truncated return value."""
    print(f"\n[{label}]")
    try:
        t0 = time.time()
        ret = fn()
        dt_ms = (time.time() - t0) * 1000
        if isinstance(ret, list):
            print(f"  status      : OK  ({dt_ms:.0f} ms)")
            print(f"  return type : list (len={len(ret)})")
            if ret:
                for x in ret[:8]:
                    print(f"    - {x}")
                if len(ret) > 8:
                    print(f"    ... +{len(ret) - 8} more")
            else:
                print(f"  ⚠️ returned EMPTY list (API may not be supported in this binary)")
        elif ret is None:
            print(f"  status      : OK ({dt_ms:.0f} ms), returned None")
        else:
            sret = repr(ret)
            if len(sret) > 200:
                sret = sret[:200] + "..."
            print(f"  status      : OK  ({dt_ms:.0f} ms)")
            print(f"  return      : {sret}")
        return ret, None
    except Exception as e:
        print(f"  status      : FAIL")
        print(f"  error       : {type(e).__name__}: {e}")
        return None, e


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=os.environ.get("AIRSIM_IP", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIRSIM_RPC_PORT", 41451)))
    parser.add_argument(
        "--candidate-assets",
        nargs="+",
        default=[
            "BP_FlySeekHumanBase",
            "BP_FlySeek_Char_03_Sophie",
            "BP_FlySeekHuman_03",
            "Sophie",
        ],
        help="Asset names to try simSpawnObject() with as smoke-spawn candidates.",
    )
    parser.add_argument(
        "--skip-spawn",
        action="store_true",
        help="跳过步骤 6 (spawn 测试) —— 仅枚举资产，安全模式（spawn 可能挂起 AirVLN）。",
    )
    parser.add_argument(
        "--substr",
        "--filter",
        dest="substr",
        default="flyseek",
        help="在 simListAssets 返回的全部资产中按子串搜索（默认 flyseek，大小写不敏感）。可写 --substr 或 --filter。",
    )
    args = parser.parse_args()

    try:
        import airsim  # type: ignore
    except ImportError as e:
        print(f"airsim not installed: {e}\n  pip install airsim==1.8.1")
        return 2

    print("=" * 72)
    print(f"AirSim Diagnostic Probe — {datetime.now().isoformat(timespec='seconds')}")
    print(f"  target = {args.ip}:{args.port}")
    print("=" * 72)

    client = airsim.MultirotorClient(ip=args.ip, port=args.port)

    # ---- 1. connectivity --------------------------------------------------
    _try("1. confirmConnection", client.confirmConnection)

    # ---- 2. vehicle pose (always supported) -------------------------------
    pose, _ = _try("2. simGetVehiclePose", client.simGetVehiclePose)

    # ---- 3. simListSceneObjects (much older, more widely supported) -------
    scene_objs, _ = _try(
        "3. simListSceneObjects()",
        lambda: client.simListSceneObjects() or [],
    )

    # ---- 4. simListAssets (newer, may not work on old binaries) -----------
    assets, _ = _try(
        "4. simListAssets()",
        lambda: client.simListAssets() or [],
    )

    # ---- 4b. substring search in full asset list --------------------------
    needle = args.substr.lower()
    matched_assets = sorted({a for a in (assets or []) if needle in a.lower()})
    print(f"\n[4b. substring search in simListAssets() for '{args.substr}' (case-insensitive)]")
    print(f"  total assets : {len(assets or [])}")
    print(f"  matches      : {len(matched_assets)}")
    for a in matched_assets[:30]:
        print(f"    - {a}")
    if len(matched_assets) > 30:
        print(f"    ... +{len(matched_assets) - 30} more")
    if not matched_assets:
        print(f"  ⚠️ 478 个资产里**没有任何**包含 '{args.substr}' 的项 → pak 八成没被 mount")

    # ---- 4c. show characters / BP-shaped assets to surface unexpected names
    bp_assets = sorted({a for a in (assets or []) if a.startswith("BP_") or a.startswith("Bp_")})
    print(f"\n[4c. all BP_* prefixed assets — 看 pak 是否注入了我们没料到的命名]")
    print(f"  total BP_ assets : {len(bp_assets)}")
    for a in bp_assets[:30]:
        print(f"    - {a}")
    if len(bp_assets) > 30:
        print(f"    ... +{len(bp_assets) - 30} more")

    # ---- 5. simListSceneObjects with FlySeek filter -----------------------
    flyseek_in_scene, _ = _try(
        "5. simListSceneObjects('.*[Ff]ly[Ss]eek.*') — regex filter",
        lambda: client.simListSceneObjects(".*[Ff]ly[Ss]eek.*") or [],
    )

    # ---- write partial report BEFORE risky spawn (in case spawn hangs) ---
    partial_out = REPO_ROOT / "flyseek_extend" / "output" / "assets" / f"diagnose_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    partial_out.parent.mkdir(parents=True, exist_ok=True)
    with partial_out.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "phase": "before_spawn",
            "scene_objects_count": len(scene_objs or []),
            "assets_count": len(assets or []),
            "flyseek_matches_in_assets": matched_assets,
            "bp_prefixed_assets": bp_assets,
            "flyseek_in_scene": list(flyseek_in_scene or []),
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[info] 已写部分报告（spawn 前）→ {partial_out}")

    successful_asset: str | None = None
    if args.skip_spawn:
        print("\n[6. SKIPPED] --skip-spawn 启用，不做 spawn 测试（安全模式）")
    elif not matched_assets and not args.candidate_assets:
        print("\n[6. SKIPPED] 步骤 4b 未找到匹配资产，且无候选名，跳过 spawn。")
    else:
        # 老版 AirSim binary 在 spawn 不存在的资产时会 hang。
        # 若步骤 4b 已经报告"没有 flyseek 资产"，强烈建议带 --skip-spawn 重跑。
        print("\n[6. simSpawnObject probe — try each candidate asset name]")
        if not matched_assets:
            print("  ⚠️ 步骤 4b 表明 pak 未加载；以下 spawn 极可能挂起。")
            print("  ⚠️ 卡 >30 秒请 Ctrl+C，并下次加 --skip-spawn")
        print("  (这是判断 pak 是否真的加载的最权威方式)")
        print("  无人机起始姿态：", pose and (pose.position.x_val, pose.position.y_val, pose.position.z_val))

        spawn_x = (pose.position.x_val if pose else 0) + 8.0
        spawn_y = (pose.position.y_val if pose else 0)
        spawn_z = (pose.position.z_val if pose else 0)
        spawn_pose = airsim.Pose(
            airsim.Vector3r(spawn_x, spawn_y, spawn_z),
            airsim.to_quaternion(0, 0, math.pi),
        )

        # 优先尝试步骤 4b 找到的真实匹配项，再退回硬编码候选
        spawn_order = matched_assets + [c for c in args.candidate_assets if c not in matched_assets]
        for cand in spawn_order:
            obj_name = f"diag_{cand}_{int(time.time())}"
            print(f"\n  → trying asset='{cand}' (is_blueprint=True)")
            try:
                t0 = time.time()
                ret = client.simSpawnObject(
                    object_name=obj_name,
                    asset_name=cand,
                    pose=spawn_pose,
                    scale=airsim.Vector3r(1.0, 1.0, 1.0),
                    physics_enabled=False,
                    is_blueprint=True,
                )
                dt = (time.time() - t0) * 1000
                print(f"    return = {ret!r}  ({dt:.0f} ms)")
                if ret:
                    successful_asset = cand
                    print(f"    ✅ SUCCESS — 实际生成对象: '{ret}'")
                    try:
                        client.simDestroyObject(ret)
                        print(f"    (已自动销毁)")
                    except Exception:
                        pass
                    break
                else:
                    print(f"    ❌ 返回空字符串")
            except Exception as e:
                print(f"    ❌ EXCEPTION: {type(e).__name__}: {e}")

    # ---- 7. summary -------------------------------------------------------
    print("\n" + "=" * 72)
    print("DIAGNOSIS SUMMARY")
    print("=" * 72)
    print(f"connectivity              : {'OK' if pose else 'BROKEN'}")
    print(f"simListSceneObjects works : {'YES' if scene_objs else 'NO (empty)'}")
    print(f"   scene_objs count       : {len(scene_objs or [])}")
    print(f"simListAssets works       : {'YES' if assets else 'NO (empty — known issue on old AirVLN)'}")
    print(f"   assets count           : {len(assets or [])}")
    print(f"FlySeek visible in scene  : {len(flyseek_in_scene or [])}")
    if flyseek_in_scene:
        for x in flyseek_in_scene[:5]:
            print(f"   - {x}")
    if args.skip_spawn:
        print(f"spawn-test                : SKIPPED (--skip-spawn)")
    else:
        print(f"spawn-test succeeded with : {successful_asset or 'NONE'}")
    print()

    if args.skip_spawn:
        if matched_assets:
            print(f"=> ✅ 发现 {len(matched_assets)} 个 FlySeek 资产，pak 加载正常。")
            print(f"   下一步去掉 --skip-spawn 跑一遍，让 spawn 验证生成图像：")
            print(f"   python scripts/diagnose_airsim.py")
        else:
            print("=> ❌ simListAssets 工作但 0 个 FlySeek 资产 → pak 没有被 UE 加载。")
            print("   排查清单：")
            print("     1) 部署的 pak 是不是最新版？用下面命令验证内部结构：")
            print("        strings <pak文件> | grep -cE 'Animations/|Blueprints/|Characters/'")
            print("        返回 ≥1 表示 pak 内有子文件夹（正确）；返回 0 是旧的废 pak")
            print("     2) AirVLN 启动后是否重启过？修改 ~mods/ 必须重启进程")
            print("     3) pak 文件名是否以 _P.pak 结尾（区分大小写）")
    elif successful_asset:
        print("=> ✅ pak overlay 工作正常。可直接用以下命令跑 smoke test：")
        print(f"   python scripts/smoke_test_spawn_and_capture.py --asset '{successful_asset}'")
    else:
        print("=> ❌ 所有候选资产都 spawn 失败。可能原因：")
        print("     a) pak 文件没被 UE 识别（检查文件名是否以 _P.pak 结尾，并重启 AirVLN）")
        print("     b) pak 内 BP 类名与候选名都对不上 —— 用 UnrealPak.exe -List 看真实名字")
        print("     c) cook 出问题导致 BP 类未导出 —— 回 UE §10.2 重新 Cook + §10.4 重新打包")
    print("=" * 72)

    # write JSON report
    out = REPO_ROOT / "flyseek_extend" / "output" / "assets" / f"diagnose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "connectivity": pose is not None,
            "scene_objects_count": len(scene_objs or []),
            "scene_objects_sample": list(scene_objs or [])[:20],
            "assets_count": len(assets or []),
            "assets_sample": list(assets or [])[:20],
            "flyseek_in_scene": list(flyseek_in_scene or []),
            "successful_spawn_asset": successful_asset,
            "candidates_tried": args.candidate_assets,
        }, f, indent=2, ensure_ascii=False)
    print(f"Full report: {out}")
    return 0 if successful_asset else 1


if __name__ == "__main__":
    sys.exit(main())
