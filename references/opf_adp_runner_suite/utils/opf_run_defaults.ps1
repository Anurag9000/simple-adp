function Set-OpfDefaultEnv {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$Value
  )

  $existing = Get-Item -Path "Env:$Name" -ErrorAction SilentlyContinue
  if ($null -eq $existing -or [string]::IsNullOrWhiteSpace($existing.Value)) {
    Set-Item -Path "Env:$Name" -Value $Value
  }
}

Set-OpfDefaultEnv -Name "PYTORCH_CUDA_ALLOC_CONF" -Value "expandable_segments:True,max_split_size_mb:128"
Set-OpfDefaultEnv -Name "CUDA_VISIBLE_DEVICES" -Value "0"
Set-OpfDefaultEnv -Name "TABULAR_CHILD_SHARED_CPU" -Value "0"
Set-OpfDefaultEnv -Name "TABULAR_CPU_WORKERS" -Value "0"
Set-OpfDefaultEnv -Name "CONCURRENCY" -Value "0"
Set-OpfDefaultEnv -Name "CONCURRENCY_FILE" -Value ""
Set-OpfDefaultEnv -Name "JOB_START_INDEX" -Value "0"
Set-OpfDefaultEnv -Name "JOB_LIMIT" -Value "0"
Set-OpfDefaultEnv -Name "MAX_ACTIVE_JOBS" -Value "0"
Set-OpfDefaultEnv -Name "MAX_ACTIVE_GPU_JOBS" -Value "0"
Set-OpfDefaultEnv -Name "REPEAT_COUNT" -Value "5"
Set-OpfDefaultEnv -Name "MAX_RETRIES" -Value "1"
Set-OpfDefaultEnv -Name "LAUNCH_HOST_RESERVE_MIB" -Value "0"
Set-OpfDefaultEnv -Name "LAUNCH_GPU_RESERVE_MIB" -Value "0"
Set-OpfDefaultEnv -Name "GPU_CHILD_GATE_CLOSE_BUFFER_MIB" -Value "100"
Set-OpfDefaultEnv -Name "HOST_CHILD_GATE_CLOSE_BUFFER_MIB" -Value "500"
Set-OpfDefaultEnv -Name "EXTERNAL_MEMORY_DROP_REOPEN_MIB" -Value "500"
Set-OpfDefaultEnv -Name "GPU_DEVICE_INDEX" -Value "0"
Set-OpfDefaultEnv -Name "CPU_ONLY" -Value "0"
Set-OpfDefaultEnv -Name "OPF_ADP_DATA_ROOT" -Value "data"
Set-OpfDefaultEnv -Name "OPF_ADP_DATA_CHUNK_SIZE" -Value "512"
Set-OpfDefaultEnv -Name "OPF_ADP_AUTO_RESUME_BUNDLE" -Value "1"
Set-OpfDefaultEnv -Name "MAX_WIDTH" -Value "512"
Set-OpfDefaultEnv -Name "MAX_DEPTH" -Value "5"
Set-OpfDefaultEnv -Name "MAX_NEURONS" -Value "1000000000"
Set-OpfDefaultEnv -Name "MAX_EPOCHS_VANILLA" -Value "0"
Set-OpfDefaultEnv -Name "MAX_EPOCHS_ADP" -Value "0"
Set-OpfDefaultEnv -Name "HOST_RAM_STOP_AVAIL_MIB" -Value "1024"
Set-OpfDefaultEnv -Name "HOST_RAM_RESUME_AVAIL_MIB" -Value "1280"
Set-OpfDefaultEnv -Name "GPU_STOP_AVAIL_MIB" -Value "100"
Set-OpfDefaultEnv -Name "GPU_RESUME_AVAIL_MIB" -Value "500"
Set-OpfDefaultEnv -Name "PRESSURE_POLL_INTERVAL_SEC" -Value "0.5"
Set-OpfDefaultEnv -Name "POST_LAUNCH_SAMPLE_DELAY_SEC" -Value "30"
Set-OpfDefaultEnv -Name "BATCH_BACKOFF_FACTOR" -Value "0.5"
Set-OpfDefaultEnv -Name "SWAP_PRESSURE_LIMIT_PCT" -Value "70"
Set-OpfDefaultEnv -Name "SWAP_RESUME_PCT" -Value "70"
Set-OpfDefaultEnv -Name "PARAM_BAND" -Value ""
Set-OpfDefaultEnv -Name "LEGACY_ARCHITECTURE_GRID" -Value "0"
Set-OpfDefaultEnv -Name "INCLUDE_SMALL_STL" -Value "0"
Set-OpfDefaultEnv -Name "INCLUDE_VANILLA" -Value "1"
Set-OpfDefaultEnv -Name "MISSING_PRESENT_TASK_REPEATS" -Value "1"
Set-OpfDefaultEnv -Name "PREDICTION_REPEATS" -Value "1"
Set-OpfDefaultEnv -Name "SURFACE" -Value "all"
Set-OpfDefaultEnv -Name "TASKS" -Value "all"
Set-OpfDefaultEnv -Name "WIDTH_ONLY_DEPTHS" -Value "1,2,3,4,5"
Set-OpfDefaultEnv -Name "WIDTH_TO_DEPTH_WIDTH" -Value "1"
Set-OpfDefaultEnv -Name "WIDTH_TO_DEPTH_DEPTH" -Value "1"
Set-OpfDefaultEnv -Name "WIDTH_TO_DEPTH_WIDTHS" -Value "1"
Set-OpfDefaultEnv -Name "DEPTH_ONLY_WIDTHS" -Value "1,2,4,8,16,32,64,128,256,512"
Set-OpfDefaultEnv -Name "HEAD_VARIANTS" -Value "single_head,two_heads,four_heads"
