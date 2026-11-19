# Source after the registry mapping blocks. Keeps patients; unresolved same-date PS is missing.
if (!length(parts)) stop("No PS sources configured")
if (dir.exists(output_dir)) stop("Use a fresh output directory: ", output_dir)
all_ps <- do.call(rbind, parts)
all_ps$quality_registry_ps_conflict <- FALSE
if (!nrow(all_ps)) stop("No supplied PS records overlap the frozen cohort")
value_counts <- tapply(all_ps$PS, all_ps$subject_id, function(x) length(unique(x)))
conflict_ids <- names(value_counts)[value_counts > 1]
if (!conflict_policy %in% c("reject", "earliest_baseline")) stop("Unknown conflict_policy")
if (conflict_policy == "earliest_baseline" && length(source_priority))
  stop("Choose earliest_baseline OR source_priority, not both")
if (length(conflict_ids) && !length(source_priority) && conflict_policy == "reject") {
  dir.create(dirname(output_dir), recursive=TRUE, showWarnings=FALSE)
  conflict_path <- paste0(output_dir, "_ps_conflicts_", format(Sys.time(), "%Y%m%d_%H%M%S"), "_INTERNAL.parquet")
  write_parquet(all_ps[all_ps$subject_id %in% conflict_ids, ],
                conflict_path)
  stop("Cross-registry PS disagreement for ", length(conflict_ids),
       " patients; inspect ", conflict_path, " and specify a justified priority or correct the source")
}
if (length(conflict_ids) && conflict_policy == "earliest_baseline") {
  conflicts <- all_ps[all_ps$subject_id %in% conflict_ids, , drop=FALSE]
  if (anyNA(conflicts$quality_registry_ps_date))
    stop("Earliest-baseline selection requires dates for every source record of conflicting patients")
  earliest <- ave(as.numeric(conflicts$quality_registry_ps_date), conflicts$subject_id, FUN=min)
  first <- conflicts[as.numeric(conflicts$quality_registry_ps_date) == earliest, , drop=FALSE]
  tied <- tapply(first$PS, first$subject_id, function(x) length(unique(x)))
  if (any(tied > 1)) {
    bad_ids <- names(tied)[tied > 1]
    audit <- first[first$subject_id %in% bad_ids, , drop=FALSE]
    audit$cohort_fine <- cohort$cohort_fine[match(audit$subject_id, cohort$subject_id)]
    dir.create(dirname(output_dir), recursive=TRUE, showWarnings=FALSE)
    audit_path <- paste0(output_dir, "_earliest_date_ties_",
                         format(Sys.time(), "%Y%m%d_%H%M%S"), "_INTERNAL.parquet")
    write_parquet(audit, audit_path)
    hit <- first$subject_id %in% bad_ids
    first$PS[hit] <- NA_real_
    first$quality_registry_ps_conflict[hit] <- TRUE
    first$quality_registry_ps_source[hit] <- "Unresolved earliest-date PS conflict"
    message("Keeping ", length(bad_ids), " patients with conflicting earliest-date PS as missing. Audit: ", audit_path)
  }
  # Remaining ties agree, or are explicitly missing due to conflicting PS.
  first <- first[order(first$subject_id, first$quality_registry_ps_source), , drop=FALSE]
  first <- first[!duplicated(first$subject_id), , drop=FALSE]
  all_ps <- rbind(all_ps[!all_ps$subject_id %in% conflict_ids, , drop=FALSE], first)
}
if (length(source_priority)) {
  if (anyDuplicated(source_priority) || !setequal(source_priority, names(parts)))
    stop("source_priority must list every loaded source exactly once")
  all_ps <- all_ps[order(match(all_ps$quality_registry_ps_source, source_priority)), ]
}
selected <- all_ps[!duplicated(all_ps$subject_id), ]
dir.create(output_dir, recursive=TRUE)
write_parquet(selected, file.path(output_dir, "quality_ps_INTERNAL.parquet"))
coverage <- merge(cohort[c("subject_id", "split", "cohort_fine")], selected,
                  by="subject_id", all.x=TRUE, sort=FALSE)
coverage$has_refreshed_ps <- !is.na(coverage$PS)
coverage$quality_registry_ps_source[is.na(coverage$quality_registry_ps_source)] <- "No refreshed PS"
counts <- aggregate(list(n_patients=rep(1L, nrow(coverage))),
                    coverage[c("split", "cohort_fine", "quality_registry_ps_source", "has_refreshed_ps")], sum)
write.csv(counts, file.path(output_dir, "quality_ps_coverage.csv"), row.names=FALSE)
write_json(list(all_required_sources_loaded=all_required_sources_loaded,
  sources=source_notes, source_priority=source_priority,
  conflict_policy=conflict_policy,
  cross_registry_conflicting_patients=length(conflict_ids),
  frozen_patients=nrow(cohort), patients_with_quality_record=nrow(selected),
  patients_with_refreshed_ps=sum(!is.na(selected$PS)),
  patients_with_unresolved_ps_conflict=sum(selected$quality_registry_ps_conflict),
  scope="PS only; other Table 1 quality-registry fields still require explicit mappings"),
  file.path(output_dir, "quality_registry_manifest.json"), pretty=TRUE, auto_unbox=TRUE)
cat("Prepared PS metadata:", file.path(output_dir, "quality_ps_INTERNAL.parquet"), "\n")
cat("All intended sources loaded:", all_required_sources_loaded, "\n")
cat("Keep INTERNAL files on the server; download only aggregate coverage/review outputs.\n")
