mod utils;
mod cache;
mod processing;

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::io::BufWriter;
use std::time::{Instant, Duration};

use cache::CacheManager;
use utils::{
    read_timstof_data, build_indexed_data, read_parquet_with_polars,
    library_records_to_dataframe, merge_library_and_report, get_unique_precursor_ids, 
    process_library_fast, create_rt_im_dicts, build_lib_matrix, build_precursors_matrix_step1, 
    build_precursors_matrix_step2, build_range_matrix_step3, build_precursors_matrix_step3, 
    build_frag_info, LibCols, PrecursorLibData, prepare_precursor_lib_data,
    extract_unique_rt_im_values, save_unique_values_to_files, UniqueValues
};
use processing::{
    FastChunkFinder, build_rt_intensity_matrix_optimized, prepare_precursor_features,
    calculate_mz_range, extract_ms2_data, build_mask_matrices, extract_aligned_rt_values,
    reshape_and_combine_matrices, process_single_precursor_rsm
};

use rayon::prelude::*;
use std::{error::Error, path::Path, env, fs::File};
use ndarray::{Array1, Array2, Array3, Array4, s, Axis};
use polars::prelude::*;
use ndarray_npy::{NpzWriter, write_npy};

// Performance monitoring structures
#[derive(Debug)]
struct ThreadMetrics {
    thread_id: usize,
    start_time: Instant,
    end_time: Instant,
    items_processed: usize,
}

#[derive(Debug)]
struct BatchPerformanceStats {
    batch_idx: usize,
    total_duration: Duration,
    per_thread_metrics: Vec<ThreadMetrics>,
    memory_usage_before: u64,
    memory_usage_after: u64,
    cpu_usage_percent: f32,
    cache_misses_estimate: u64,
}

// New struct to hold RSM results
#[derive(Debug)]
pub struct RSMPrecursorResults {
    pub index: usize,
    pub precursor_id: String,
    pub rsm_matrix: Array4<f32>,  // Shape: [1, 5, 72, 396]
    pub all_rt: Vec<f32>,          // 396 RT values
}

// Helper function to get current memory usage
fn get_memory_usage() -> u64 {
    use sysinfo::{System, SystemExt};
    let mut sys = System::new();
    sys.refresh_memory();
    sys.used_memory() * 1024 // Convert to bytes
}

// Helper function to estimate CPU usage
fn get_cpu_usage() -> f32 {
    use sysinfo::{System, SystemExt, ProcessExt};
    let mut sys = System::new();
    sys.refresh_processes();
    
    if let Some(process) = sys.process(sysinfo::get_current_pid().unwrap()) {
        process.cpu_usage()
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Fixed parameters
    let batch_size = 1000;
    let parallel_threads = std::env::var("PARALLEL_THREADS")
        .unwrap_or_else(|_| "64".to_string())
        .parse::<usize>()
        .unwrap_or(64);
    let output_dir = "output_test";
    
    let d_folder = "/wangshuaiyao/dia-bert-timstof/test_data/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    let report_file_path = "/wangshuaiyao/dia-bert-timstof/lib/20250730_v5.3_TPHPlib_frag1025_swissprot_final_all_from_Yueliang.parquet";
    let lib_file_path = "/wangshuaiyao/dia-bert-timstof/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang_with_decoy.tsv";
    
    // Initialize thread pool and print system info
    rayon::ThreadPoolBuilder::new()
        .num_threads(parallel_threads)
        .build_global()
        .unwrap();
    
    println!("\n========== SYSTEM CONFIGURATION ==========");
    println!("Parallel threads configured: {}", parallel_threads);
    println!("Available CPU cores: {}", num_cpus::get());
    println!("Available logical CPUs: {}", num_cpus::get_physical());
    println!("Initial memory usage: {:.2} GB", get_memory_usage() as f64 / 1e9);
    println!("Batch size: {}", batch_size);
    println!("Output directory: {}", output_dir);
    
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }
    
    // Create output directory
    std::fs::create_dir_all(&output_dir)?;
    
    // ================================ DATA LOADING AND INDEXING ================================
    let cache_manager = CacheManager::new();
    
    println!("\n========== DATA PREPARATION PHASE ==========");
    let total_start = Instant::now();
        
    let (ms1_indexed, ms2_indexed_pairs) = if cache_manager.is_cache_valid(d_path) {
        println!("Found valid cache, loading indexed data directly...");
        let cache_load_start = Instant::now();
        let (ms1_indexed, ms2_indexed_pairs) = cache_manager.load_indexed_data(d_path)?;
        println!("Cache loading time: {:.5} seconds", cache_load_start.elapsed().as_secs_f32());
        (ms1_indexed, ms2_indexed_pairs)
    } else {
        println!("Cache invalid or non-existent, reading TimsTOF data...");
        
        let raw_data_start = Instant::now();
        let raw_data = read_timstof_data(d_path)?;
        println!("Raw data reading time: {:.5} seconds", raw_data_start.elapsed().as_secs_f32());
        println!("  - MS1 data points: {}", raw_data.ms1_data.mz_values.len());
        println!("  - MS2 windows: {}", raw_data.ms2_windows.len());
        
        println!("\nBuilding indexed data structures...");
        let index_start = Instant::now();
        let (ms1_indexed, ms2_indexed_pairs) = build_indexed_data(raw_data)?;
        println!("Index building time: {:.5} seconds", index_start.elapsed().as_secs_f32());
        
        let cache_save_start = Instant::now();
        cache_manager.save_indexed_data(d_path, &ms1_indexed, &ms2_indexed_pairs)?;
        println!("Cache saving time: {:.5} seconds", cache_save_start.elapsed().as_secs_f32());
        
        (ms1_indexed, ms2_indexed_pairs)
    };
    
    println!("Total data preparation time: {:.5} seconds", total_start.elapsed().as_secs_f32());
    
    let finder = FastChunkFinder::new(ms2_indexed_pairs)?;
    
    // ================================ LIBRARY AND REPORT LOADING ================================
    println!("\n========== LIBRARY AND REPORT PROCESSING ==========");
    let lib_processing_start = Instant::now();

    let library_records = process_library_fast(lib_file_path)?;
    let library_df = library_records_to_dataframe(library_records.clone())?;

    let report_df = read_parquet_with_polars(report_file_path)?;
    let diann_result = merge_library_and_report(library_df, report_df)?;
    
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    println!("diann_precursor_id_all: {:?}", diann_precursor_id_all.head(Some(5)));
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    println!("Library and report processing time: {:.5} seconds", lib_processing_start.elapsed().as_secs_f32());
    
    let device = "cpu";
    let frag_repeat_num = 5;
    
    // ================================ BATCH PRECURSOR PROCESSING ================================
    println!("\n========== BATCH PRECURSOR PROCESSING ==========");
    
    println!("\n[Step 1] Preparing library data for all precursors");
    let prep_start = Instant::now();
    
    let unique_precursor_ids: Vec<String> = diann_precursor_id_all
        .column("transition_group_id")?
        .str()?
        .into_iter()
        .filter_map(|opt| opt.map(|s| s.to_string()))
        .collect();

    let total_unique_precursors = unique_precursor_ids.len();
    println!("\n========== LIBRARY STATISTICS ==========");
    println!("Total unique precursor IDs in library: {}", total_unique_precursors);
    
    let lib_cols = LibCols::default();
    
    // Process all precursors (no max_precursors limit)
    let precursor_lib_data_list = prepare_precursor_lib_data(
        &library_records,
        &unique_precursor_ids,
        &assay_rt_kept_dict,
        &assay_im_kept_dict,
        &lib_cols,
        total_unique_precursors,
    )?;
    
    println!("  - Prepared data for {} precursors", precursor_lib_data_list.len());
    println!("  - Preparation time: {:.5} seconds", prep_start.elapsed().as_secs_f32());
    
    drop(library_records);
    println!("  - Released library_records from memory");
    
    // Process in batches
    let total_batches = (precursor_lib_data_list.len() + batch_size - 1) / batch_size;
    println!("\n[Step 2] Processing {} precursors in {} batches", 
             precursor_lib_data_list.len(), total_batches);
    
    // Global performance tracking - specify type explicitly
    let mut all_batch_stats: Vec<BatchPerformanceStats> = Vec::new();
    
    for batch_idx in 0..total_batches {
        let batch_start_idx = batch_idx * batch_size;
        let batch_end_idx = ((batch_idx + 1) * batch_size).min(precursor_lib_data_list.len());
        let batch_precursors = &precursor_lib_data_list[batch_start_idx..batch_end_idx];
        
        println!("\n========== Processing Batch {}/{} ==========", batch_idx + 1, total_batches);
        println!("Precursors {} to {} (total: {})", 
                 batch_start_idx + 1, batch_end_idx, batch_precursors.len());
        
        let batch_start = Instant::now();
        let memory_before = get_memory_usage();
        
        // Performance tracking for this batch
        let processed_count = Arc::new(AtomicUsize::new(0));
        let thread_start_times = Arc::new(Mutex::new(Vec::<Instant>::new()));
        let thread_end_times = Arc::new(Mutex::new(Vec::<Instant>::new()));
        let thread_work_distribution = Arc::new(Mutex::new(std::collections::HashMap::<usize, usize>::new()));
        
        let batch_count = batch_precursors.len();
        let results_mutex = Arc::new(Mutex::new(Vec::new()));
        
        // Track work stealing and thread utilization
        let work_steal_count = Arc::new(AtomicU64::new(0));
        let total_processing_time = Arc::new(AtomicU64::new(0));
        
        // Process batch in parallel with detailed monitoring
        batch_precursors
            .par_iter()
            .enumerate()
            .for_each(|(batch_internal_idx, precursor_data)| {
                let thread_id = rayon::current_thread_index().unwrap_or(999);
                let task_start = Instant::now();
                
                // Record thread start
                {
                    let mut starts = thread_start_times.lock().unwrap();
                    if starts.len() <= thread_id {
                        starts.resize(thread_id + 1, Instant::now());
                    }
                    starts[thread_id] = task_start;
                }
                
                let global_index = batch_start_idx + batch_internal_idx;
                
                let result = process_single_precursor_rsm(
                    precursor_data,
                    &ms1_indexed,
                    &finder,
                    frag_repeat_num,
                    device,
                );
                
                let task_duration = task_start.elapsed();
                total_processing_time.fetch_add(task_duration.as_micros() as u64, Ordering::Relaxed);
                
                // Track work distribution
                {
                    let mut dist = thread_work_distribution.lock().unwrap();
                    *dist.entry(thread_id).or_insert(0) += 1;
                }
                
                let current = processed_count.fetch_add(1, Ordering::SeqCst) + 1;
                
                // Log every 100 items for thread distribution
                if current % 100 == 0 {
                    println!("[Thread {}] Processed {} items, last item took {:.3}ms", 
                             thread_id, current, task_duration.as_secs_f32() * 1000.0);
                }
                
                match result {
                    Ok((precursor_id, rsm_matrix, all_rt)) => {
                        let rsm_result = RSMPrecursorResults {
                            index: batch_internal_idx,
                            precursor_id: precursor_id.clone(),
                            rsm_matrix,
                            all_rt,
                        };
                        
                        let mut results = results_mutex.lock().unwrap();
                        results.push(rsm_result);
                    },
                    Err(e) => {
                        eprintln!("[Thread {} - Batch {} - {}/{}] ✗ Error processing {} (global index: {}): {}", 
                                  thread_id, batch_idx + 1, current, batch_count, 
                                  precursor_data.precursor_id, global_index, e);
                    }
                }
            });
        
        let batch_elapsed = batch_start.elapsed();
        let memory_after = get_memory_usage();
        let cpu_usage = get_cpu_usage();
        
        // Analyze thread work distribution
        let work_dist = thread_work_distribution.lock().unwrap();
        let mut thread_counts: Vec<(usize, usize)> = work_dist.iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        thread_counts.sort_by_key(|&(k, _)| k);
        
        println!("\n========== THREAD UTILIZATION ANALYSIS ==========");
        println!("Thread work distribution:");
        for (thread_id, count) in &thread_counts {
            println!("  Thread {}: {} items ({:.1}%)", 
                     thread_id, count, (*count as f32 / batch_count as f32) * 100.0);
        }
        
        let max_work = thread_counts.iter().map(|(_, c)| c).max().unwrap_or(&0);
        let min_work = thread_counts.iter().map(|(_, c)| c).min().unwrap_or(&0);
        let work_imbalance = if *min_work > 0 {
            (*max_work as f32 / *min_work as f32) - 1.0
        } else {
            f32::INFINITY
        };
        
        println!("\nWork imbalance factor: {:.2}x", work_imbalance + 1.0);
        println!("Active threads: {} / {}", thread_counts.len(), parallel_threads);
        
        let total_proc_time = total_processing_time.load(Ordering::Relaxed) as f64 / 1e6;
        let theoretical_speedup = total_proc_time / batch_elapsed.as_secs_f64();
        println!("Theoretical speedup: {:.2}x (total work: {:.2}s, elapsed: {:.2}s)", 
                 theoretical_speedup, total_proc_time, batch_elapsed.as_secs_f64());
        
        println!("\n========== MEMORY ANALYSIS ==========");
        println!("Memory before batch: {:.2} GB", memory_before as f64 / 1e9);
        println!("Memory after batch: {:.2} GB", memory_after as f64 / 1e9);
        println!("Memory delta: {:.2} MB", (memory_after as i64 - memory_before as i64) as f64 / 1e6);
        println!("CPU usage: {:.1}%", cpu_usage);
        
        println!("\n========== SAVING BATCH {} RESULTS ==========", batch_idx + 1);
        let save_start = Instant::now();
        
        let mut results = Arc::try_unwrap(results_mutex).unwrap().into_inner().unwrap();
        
        // Sort results by original index to restore order
        results.sort_by_key(|r| r.index);
        
        // Save batch results
        save_batch_results_as_npy(&results, batch_precursors, batch_idx, &output_dir)?;
        
        println!("Batch {} save time: {:.5} seconds", batch_idx + 1, save_start.elapsed().as_secs_f32());
        
        println!("\n========== BATCH {} SUMMARY ==========", batch_idx + 1);
        println!("Processed: {} precursors", results.len());
        println!("Failed: {} precursors", batch_count - results.len());
        println!("Batch processing time: {:.5} seconds", batch_elapsed.as_secs_f32());
        println!("Average time per precursor: {:.5} seconds", 
                 batch_elapsed.as_secs_f32() / batch_count as f32);
        println!("Efficiency: {:.1}% (theoretical speedup / thread count)", 
                 (theoretical_speedup / parallel_threads as f64) * 100.0);
    }
    
    println!("\n========== OVERALL PROCESSING SUMMARY ==========");
    println!("Total unique precursor IDs in library: {}", total_unique_precursors);
    println!("Total processed: {} precursors", precursor_lib_data_list.len());
    println!("Processing mode: Parallel ({} threads)", parallel_threads);
    println!("Batch size: {}", batch_size);
    println!("Total batches: {}", total_batches);
    println!("Output directory: {}", output_dir);
    
    // Final performance analysis
    println!("\n========== PERFORMANCE ANALYSIS RECOMMENDATIONS ==========");
    println!("Based on the execution profile:");
    println!("1. Check thread work distribution - uneven distribution indicates load balancing issues");
    println!("2. Monitor memory delta - large increases may indicate memory pressure");
    println!("3. Compare theoretical speedup vs actual thread count to identify bottlenecks");
    println!("4. Low efficiency % suggests contention or I/O bottlenecks");
    
    Ok(())
}

fn save_batch_results_as_npy(
    results: &[RSMPrecursorResults], 
    original_precursor_list: &[PrecursorLibData],
    batch_idx: usize,
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    
    if results.is_empty() {
        return Err("No results to save".into());
    }
    
    // Create batch-specific filenames with simple numbering
    let batch_name = format!("batch_{}", batch_idx);
    let rsm_filename = format!("{}/{}_rsm.npy", output_dir, batch_name);
    let rt_filename = format!("{}/{}_rt_values.npy", output_dir, batch_name);
    let index_filename = format!("{}/{}_index.txt", output_dir, batch_name);
    
    // Create a map of index to result for quick lookup
    let mut result_map = std::collections::HashMap::new();
    for result in results {
        result_map.insert(result.index, result);
    }
    
    // Initialize the combined RSM matrix and RT values based on original order
    let n_original = original_precursor_list.len();
    let frag_repeat_num = 5;
    let n_fragments = 72; // MS1 + MS2 fragments
    let n_scans = 396;
    
    // Initialize with zeros
    let mut all_rsm_matrix = Array4::<f32>::zeros((n_original, frag_repeat_num, n_fragments, n_scans));
    let mut all_rt_values = Array2::<f32>::zeros((n_original, n_scans));
    let mut precursor_ids = Vec::with_capacity(n_original);
    let mut status_list = Vec::with_capacity(n_original);
    
    // Fill matrices in original order
    for (i, precursor_data) in original_precursor_list.iter().enumerate() {
        precursor_ids.push(precursor_data.precursor_id.clone());
        
        if let Some(result) = result_map.get(&i) {
            // Successfully processed - copy the RSM matrix
            all_rsm_matrix.slice_mut(s![i, .., .., ..]).assign(&result.rsm_matrix.slice(s![0, .., .., ..]));
            
            // Copy RT values
            for (j, &rt_val) in result.all_rt.iter().enumerate() {
                if j < n_scans {
                    all_rt_values[[i, j]] = rt_val;
                }
            }
            status_list.push("SUCCESS");
        } else {
            // Failed to process - keep as zeros
            status_list.push("FAILED");
        }
    }
    
    // Save RSM matrix
    println!("Saving RSM matrix to: {}", rsm_filename);
    println!("  Shape: [{}, {}, {}, {}]", n_original, frag_repeat_num, n_fragments, n_scans);
    write_npy(&rsm_filename, &all_rsm_matrix)?;
    
    // Save RT values
    println!("Saving RT values to: {}", rt_filename);
    println!("  Shape: [{}, {}]", n_original, n_scans);
    write_npy(&rt_filename, &all_rt_values)?;
    
    // Save index file
    println!("Saving index file to: {}", index_filename);
    let mut id_file = File::create(&index_filename)?;
    writeln!(id_file, "# Index file for RSM matrices and RT values - Batch {}", batch_idx)?;
    writeln!(id_file, "# Total precursors in batch: {}", n_original)?;
    writeln!(id_file, "# Successfully processed: {}", results.len())?;
    writeln!(id_file, "# Failed: {}", n_original - results.len())?;
    writeln!(id_file, "# RSM matrix shape: [{}, {}, {}, {}]", n_original, frag_repeat_num, n_fragments, n_scans)?;
    writeln!(id_file, "# RT values shape: [{}, {}]", n_original, n_scans)?;
    writeln!(id_file, "# Row_Index\tPrecursor_ID\tStatus")?;
    
    for (i, (id, status)) in precursor_ids.iter().zip(status_list.iter()).enumerate() {
        writeln!(id_file, "{}\t{}\t{}", i, id, status)?;
    }
    
    println!("\nSuccessfully saved batch {} files:", batch_idx);
    println!("  - RSM matrix: {}", rsm_filename);
    println!("  - RT values: {}", rt_filename);
    println!("  - Index file: {}", index_filename);
    
    Ok(())
}
