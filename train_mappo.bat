@echo off
REM ============================================================
REM  Splendor MAPPO / PPO 一键训练脚本 (Windows + RTX 5070)
REM  用法:
REM    train_mappo.bat                -> team 模式 (MAPPO/CTDE)
REM    train_mappo.bat standard       -> standard 模式 (PPO/self-play)
REM    train_mappo.bat team 5000000   -> 自定义步数
REM  查看曲线: tensorboard --logdir runs
REM ============================================================
setlocal
set MODE=%1
if "%MODE%"=="" set MODE=team
set TIMESTEPS=%2
if "%TIMESTEPS%"=="" set TIMESTEPS=2000000

set WORKERS=7
set HORIZON=2048
set ROLLOUTS=16
set MINIBATCH=32768
set EPOCHS=4
set EVAL_EVERY=10
set EVAL_GAMES=32
set SEED=1234

if "%MODE%"=="team" (
    set SAVE=checkpoints/mappo_team.pt
    set TB=runs/team
    set TARGET=30
) else (
    set SAVE=checkpoints/ppo_standard.pt
    set TB=runs/standard
    set TARGET=15
)

echo [train] mode=%MODE% timesteps=%TIMESTEPS% target=%TARGET% workers=%WORKERS% horizon=%HORIZON% rollouts=%ROLLOUTS% minibatch=%MINIBATCH%

set SDL_VIDEODRIVER=dummy
python -m rl.mappo.main_train ^
    --mode %MODE% ^
    --timesteps %TIMESTEPS% ^
    --workers %WORKERS% ^
    --target-score %TARGET% ^
    --horizon %HORIZON% ^
    --rollouts-per-update %ROLLOUTS% ^
    --minibatch-size %MINIBATCH% ^
    --epochs %EPOCHS% ^
    --eval-every %EVAL_EVERY% ^
    --eval-games %EVAL_GAMES% ^
    --seed %SEED% ^
    --save %SAVE% ^
    --tb-runs %TB%

echo [train] done. model -> %SAVE%
endlocal
