#!/bin/bash
set -euo pipefail

# Default to the image's non-root application user and the two writable paths we
# need to prepare before handing off to the real process.
orca_user="orca"
orca_home="/home/orca"
sky_state_dir="${orca_home}/.sky"
outputs_dir="/app/outputs"

# Start from the baked-in UID/GID. We only change them if the bind-mounted
# outputs directory points us at a non-root host owner.
current_uid="$(id -u "${orca_user}")"
current_gid="$(id -g "${orca_user}")"
target_uid="${current_uid}"
target_gid="${current_gid}"
id_changed=0

# Match the bind-mounted outputs directory owner when it already exists on the host.
# Outcomes:
# - outputs_dir missing: keep the image UID/GID.
# - outputs_dir owned by a non-root host user/group: adopt that UID/GID target.
# - outputs_dir owned by root: ignore UID/GID 0 so orca stays non-root.
if [ -d "${outputs_dir}" ]; then
    outputs_uid="$(stat -c '%u' "${outputs_dir}")"
    outputs_gid="$(stat -c '%g' "${outputs_dir}")"

    if [ "${outputs_uid}" != "0" ]; then
        target_uid="${outputs_uid}"
    fi

    if [ "${outputs_gid}" != "0" ]; then
        target_gid="${outputs_gid}"
    fi
fi

# Reconcile the group first so later ownership fixes can use the final group.
# Outcomes:
# - target_gid matches current_gid: nothing changes.
# - target_gid already belongs to another group: move orca into that group.
# - target_gid is free: retag the orca group with that GID.
if [ "${target_gid}" != "${current_gid}" ]; then
    existing_group="$(getent group "${target_gid}" | cut -d: -f1 || true)"
    if [ -n "${existing_group}" ] && [ "${existing_group}" != "${orca_user}" ]; then
        usermod -g "${target_gid}" "${orca_user}"
    else
        groupmod -o -g "${target_gid}" "${orca_user}"
    fi
    id_changed=1
fi

# Reconcile the user UID after the group change.
# Outcomes:
# - target_uid matches current_uid: nothing changes.
# - target_uid already belongs to another user: warn and keep orca's current UID.
# - target_uid is free: retag orca with the requested UID.
if [ "${target_uid}" != "${current_uid}" ]; then
    existing_user="$(getent passwd "${target_uid}" | cut -d: -f1 || true)"
    if [ -n "${existing_user}" ] && [ "${existing_user}" != "${orca_user}" ]; then
        printf 'warning: UID %s already belongs to %s; keeping %s at UID %s\n' \
            "${target_uid}" "${existing_user}" "${orca_user}" "${current_uid}" >&2
    else
        usermod -o -u "${target_uid}" "${orca_user}"
        id_changed=1
    fi
fi

# Capture the final effective group name in case orca was attached to an existing
# group instead of keeping the literal "orca" group name.
orca_group="$(id -gn "${orca_user}")"

# Ensure the Sky state directory always exists before we fix ownership.
install -d -m 0755 "${sky_state_dir}"

# Apply ownership fixes based on whether the identity changed.
# Outcomes:
# - UID/GID changed: recursively repair /app and /home/orca for the new identity.
# - UID/GID unchanged: only repair the state dir and outputs dir when present.
if [ "${id_changed}" = "1" ]; then
    chown -R "${orca_user}:${orca_group}" /app "${orca_home}"
else
    chown -R "${orca_user}:${orca_group}" "${sky_state_dir}"
    if [ -d "${outputs_dir}" ]; then
        chown -R "${orca_user}:${orca_group}" "${outputs_dir}"
    fi
fi

# Drop root privileges and replace the shell with the requested container command
# so the application runs as orca regardless of the setup path above.
exec gosu "${orca_user}" "$@"
