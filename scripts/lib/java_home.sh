# shellcheck shell=sh
# Resolve JAVA_HOME to a JDK 17+ that Gradle 8.7 / AGP 8.5.1 can use. Sourced, not executed:
#
#   . "$(dirname "$0")/lib/java_home.sh"
#
# Resolution order — an explicit JAVA_HOME always wins, then a JDK 17+ on PATH, then Android Studio's
# bundled JBR at its Linux default.
#
# The PATH probe exists because `/opt/android-studio/jbr` is a *Linux Android Studio* path: it is
# correct on exactly one kind of machine and simply absent on macOS, on a CI runner, or under any
# standalone JDK. Four scripts and the Makefile each hardcoded it as their only fallback, so on any
# other machine Gradle failed with a Java-version error that named neither JAVA_HOME nor the script
# that set it. Mirrors the identical probe in the Makefile; keep the two in step.
#
# On failure this sets nothing and prints what to do. It deliberately does NOT exit — the caller
# decides whether a missing JDK is fatal (`make doctor` reports it and carries on).

mt_resolve_java_home() {
  mt_jh_candidate=""

  if [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/java" ]; then
    return 0
  fi

  if command -v java >/dev/null 2>&1; then
    if java -version 2>&1 | head -1 | grep -qE '"(1[7-9]|2[0-9])'; then
      mt_jh_candidate="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
    fi
  fi

  if [ -z "$mt_jh_candidate" ] && [ -x /opt/android-studio/jbr/bin/java ]; then
    mt_jh_candidate=/opt/android-studio/jbr
  fi

  if [ -n "$mt_jh_candidate" ]; then
    JAVA_HOME="$mt_jh_candidate"
    export JAVA_HOME
    return 0
  fi

  echo "JAVA_HOME is not set and no JDK 17+ was found." >&2
  echo "  Gradle 8.7 / AGP 8.5.1 need JDK 17 or newer. Either:" >&2
  echo "    export JAVA_HOME=/path/to/jdk17        # any JDK 17+" >&2
  echo "    export JAVA_HOME=\"\$(/usr/libexec/java_home -v 17)\"   # macOS" >&2
  echo "  Android Studio ships one; on Linux it is usually /opt/android-studio/jbr." >&2
  echo "  Run 'make doctor' for the full prerequisite report." >&2
  return 1
}

mt_resolve_java_home
