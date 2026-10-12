# Sanitization report

The source release was prepared with these exclusions:

- no personal user names or home-directory paths;
- no internal host names, cluster accounts, partitions, or scheduler job IDs;
- no API keys, access tokens, proxy configuration, or telemetry endpoints;
- no training/evaluation logs, cached outputs, generated videos, or checkpoint
  tensors;
- no private prompt corpus or precomputed embeddings.

Public upstream project names and license notices are retained where required
for attribution and legal compliance. Automated release-contract tests scan the
package for private absolute paths and common credential markers.
