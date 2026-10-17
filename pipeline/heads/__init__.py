"""Moderation heads on the shared image embedding (spec §4.1 T2, §4.2): linear probes trained on
Gemini and `verify` labels, isotonic calibration, and two thresholds per head – pass below
`low`, reject at or above `high`, escalate in between.
"""
