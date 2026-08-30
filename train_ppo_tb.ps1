<#
.SYNOPSIS
在 Windows PowerShell 中启动象棋 PPO 训练。

.EXAMPLE
.\train_ppo_tb.ps1

.EXAMPLE
.\train_ppo_tb.ps1 -Opponent selfplay -Updates 1000

.EXAMPLE
.\train_ppo_tb.ps1 -Opponent model -OpponentModel .\checkpoints\frozen_pointer.pt
#>
[CmdletBinding()]
param(
    [ValidateSet("selfplay", "model", "search")]
    [string]$Opponent = "selfplay",
    [string]$OpponentModel,
    [string]$Resume,
    [int]$Updates = 1000,
    [int]$NumEnvs = 8,
    [int]$RolloutSteps = 128,
    [int]$Epochs = 4,
    [double]$LearningRate = 2.5e-4,
    [int]$SaveInterval = 25,
    [int]$SearchDepth = 3,
    [switch]$SearchQuiescence,
    [switch]$OpponentStochastic,
    [string]$Python
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $root ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "未找到虚拟环境 Python：$Python。请先执行：py -m venv .venv"
}

if ($Opponent -eq "model") {
    if (-not $OpponentModel) {
        throw "-Opponent model 时必须提供 -OpponentModel <冻结模型检查点路径>"
    }
    if (-not (Test-Path -LiteralPath $OpponentModel -PathType Leaf)) {
        throw "找不到固定对手模型：$OpponentModel"
    }
    $OpponentModel = (Resolve-Path -LiteralPath $OpponentModel).Path
}

if ($Resume) {
    if (-not (Test-Path -LiteralPath $Resume -PathType Leaf)) {
        throw "找不到恢复训练检查点：$Resume"
    }
    $Resume = (Resolve-Path -LiteralPath $Resume).Path
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $root "runs\ppo_$stamp"
$checkpoint = Join-Path $root "checkpoints\xiangqi_ppo_$stamp.pt"
New-Item -ItemType Directory -Force -Path $logDir, (Split-Path -Parent $checkpoint) | Out-Null

& $Python -m pip install -r (Join-Path $root "requirements-train.txt")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$trainArgs = @(
    "--updates", $Updates,
    "--num-envs", $NumEnvs,
    "--rollout-steps", $RolloutSteps,
    "--epochs", $Epochs,
    "--learning-rate", $LearningRate,
    "--save-interval", $SaveInterval,
    "--opponent", $Opponent,
    "--search-depth", $SearchDepth,
    "--log-dir", $logDir,
    "--output", $checkpoint
)
if ($SearchQuiescence) {
    $trainArgs += "--search-quiescence"
}
if ($Opponent -eq "model") {
    $trainArgs += "--opponent-model", $OpponentModel
    if ($OpponentStochastic) {
        $trainArgs += "--opponent-stochastic"
    }
}
if ($Resume) {
    $trainArgs += "--resume", $Resume
}

Write-Host "TensorBoard 日志：$logDir"
Write-Host "模型检查点：$checkpoint"
Write-Host "对手模式：$Opponent"
if ($Opponent -eq "model") {
    Write-Host "固定对手模型：$OpponentModel"
}
if ($Resume) {
    Write-Host "恢复模型：$Resume"
} else {
    Write-Host "恢复模型：无（从头训练）"
}
Write-Host "TensorBoard：$Python -m tensorboard.main --logdir $root\runs --port 6006"

& $Python -m rl.train @trainArgs
exit $LASTEXITCODE
