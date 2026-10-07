## Secret scanning

GitHub Actions runs the pinned Infisical CLI on new push and pull-request commits.
No Infisical login or application credentials are required. Findings fail the check;
the job summary lists locations and rules without matched source or secret values.
Use **Actions → Secret scanning → Run workflow** for a current-files scan, or enable
`full_history` to audit all Git history. There is no schedule or automatic retry.
Rotate real exposed credentials before removing them; review fixture false positives
individually. Results stay in GitHub, outside Infisical's Findings dashboard.
