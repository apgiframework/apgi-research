# APGI Operational Playbooks

**Last Updated:** April 2026

---

## 1. Incident Response Playbook

**Scope:** Responding to APGI system crashes, LLM integration failures, or security boundary violations.

**Steps:**

1. **Triage**: Check `apgi_logging` outputs to identify the faulting module (Engine, Validator, Security Adapter).
2. **Containment**: If an untrusted execution boundary is breached, halt the experiment queue immediately and lock the environment via `get_authz_manager().emergency_lock(reason=...)` (deactivates all operators, revokes all tokens; lift with `emergency_unlock(operator_ids=[...])`).
3. **Investigation**: Query the `get_audit_sink()` audit trails to identify the latest parameter changes or experiment states leading to the crash.

   ```python
   from apgi_audit import get_audit_sink
   audit = get_audit_sink()
   events = audit.get_events(limit=50)
   ```

4. **Resolution**: Roll back to the last known robust state using git and restart via `cli_entrypoint()`.

---

## 2. Timeout Tuning Playbook

**Scope:** Handling `TimeoutError` in asynchronous agents and long-running batch processing.

**Steps:**

1. **Metrics Review**: Inspect execution times from `apgi_profiler.get_metrics()` or `XPRAgentEngineEnhanced.analyze_performance_trend()`.
2. **Adjustments**: If baseline latency has grown, use `get_timeout_manager()` to increase bounds:

   ```python
   from apgi_timeout_abstraction import get_timeout_manager
   manager = get_timeout_manager()
   with manager.timeout(new_timeout_seconds, "Operation description"):
       run_experiment()
   ```

3. **Validation**: Test temporal parameters in isolation using `tests/test_apgi_timeout_abstraction.py`.

---

## 3. Performance Regressions Playbook

**Scope:** Identifying drops in evaluation matrices or spikes in memory allocation.

**Steps:**

1. **Profiling**: Enable `APGI_ENABLE_PROFILING=true` and inspect hotspots via `apgi_profiler` output.
2. **Memory Leaks**: Ensure all new history logs use bounded structures (`collections.deque` with maxlen).
3. **Re-baseline**: Document new baseline overhead in `docs/Maturity-Matrix.md` and update benchmark gates.

---

## 4. Security Incident Playbook

**Scope:** Responding to security violations (unauthorized access, audit integrity failures).

**Steps:**

1. **Detection**: Monitor `get_security_factory().get_metrics()` for denied operations:

   ```python
   from apgi_security_adapters import get_security_factory
   metrics = get_security_factory().get_metrics()
   print(f"Denied: {metrics['total_denied']}")
   ```

2. **Audit Verification**: Verify audit trail integrity:

   ```python
   from apgi_audit import get_audit_sink
   if not get_audit_sink().verify_integrity():
       alert_security_team()
   ```

3. **Containment**: Revoke operator access via `get_authz_manager().revoke_operator(operator_id)`.
4. **Report**: Export audit trail for investigation: `audit.export_audit_trail("incident.json")`.

---

## 5. Compliance Audit Playbook

**Scope:** Preparing for GDPR/CCPA/HIPAA compliance audits.

**Steps:**

1. **Data Classification Review**: Verify all experiments have proper classification:

   ```python
   from apgi_compliance import get_compliance_manager
   report = get_compliance_manager().generate_compliance_report()
   ```

2. **Retention Policy Check**: Review pending deletions:

   ```python
   from apgi_data_retention import get_retention_scheduler
   stats = get_retention_scheduler().get_retention_statistics()
   print(f"Pending deletion: {stats['pending_deletion']}")
   ```

3. **Audit Export**: Generate audit trail exports for compliance reviewers:

   ```python
   get_audit_sink().export_audit_trail("gdpr_audit_trail.json")
   ```

4. **Documentation**: Ensure `docs/APGI-COMPLIANCE.md` control matrix is up to date.

---

## 6. Configuration Drift Playbook

**Scope:** Resolving configuration inconsistencies across environments.

**Steps:**

1. **Source Check**: Use `get_config().get_all_sources()` to identify config value origins.
2. **Environment Audit**: Verify `APGI_*` environment variables are consistent:

   ```bash
   env | grep APGI_
   ```

3. **Validation**: Run config validation: `python -c "from apgi_config import get_config; get_config().get_security_config()"`
4. **Reload**: Force config reload: `get_config().reload()`

---

## Quick Reference: Module Emergency Contacts

| Module | Function | Emergency Check |
| ------ | -------- | ----------------- |
| `apgi_audit` | `get_audit_sink().verify_integrity()` | Audit chain broken |
| `apgi_authz` | `get_authz_manager().list_active_operators()` | Unauthorized access |
| `apgi_security_adapters` | `get_security_factory().get_metrics()` | Security violations |
| `apgi_config` | `get_config().get_all_sources()` | Config drift |
| `apgi_timeout_abstraction` | `get_timeout_manager()` | Timeout failures |
| `apgi_data_retention` | `get_retention_scheduler().get_retention_statistics()` | Data retention issues |
