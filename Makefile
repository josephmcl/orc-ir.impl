# Makefile for orc-ir cross-platform code
#
#
#
# ----------------------------------------------------------------- #
# Build envrionment variables
hostname = $(shell hostname -f)

environment_directory      := ./.configure/environment
toolkit_makefile_directory := build
target_directory           := target
source_directory           := source
include_directory          := include
object_directory           := object

environment_file = $(environment_directory)/$(hostname)
ifneq ("$(wildcard $(environment_file))","")
    environment_exists := 1
	-include $(environment_file)
	device := ${DEVICE}
	toolkit := ${TOOLKIT}
	toolkit_version := ${TOOLKIT_VERSION}
	environment_slug := $(device).$(toolkit).$(toolkit_version)
else
    environment_exists := 0
endif

toolkit_makefile_file = \
	$(toolkit_makefile_directory)/$(toolkit).$(device)
ifneq ("$(wildcard $(toolkit_makefile_file))","")
	toolkit_makefile_exists := 1
	-include $(toolkit_makefile_file)
else
	toolkit_makefile_exists := 0
endif

target = $(target_directory)/$(environment_slug)/$(target_file)

.DEFAULT_GOAL := all
# ----------------------------------------------------------------- #
# Pretty printing and error reporting.

# Ascii color variables
esc   := $(shell printf '\033')
blue  := $(esc)[34m
red   := $(esc)[31m
reset := $(esc)[0m
# Newline variable
define endl


endef
# Status function
define status
$(info $(blue)(make)$(reset)$(1))
endef
# error funtction
define failure
$(info $(red)(error)$(reset)$(1))
@exit 2
endef
# ----------------------------------------------------------------- #

all: environment build-rules generate solve symlinks

# ----------------------------------------------------------------- #
.PHONY: symlinks
symlinks: generate solve
	@ln -sf $(generate_bin) ./generate
	@ln -sf $(solve_bin) ./solve
	$(call status, Synchronized symbolic links to latest targets.\
	$(endl)$(endl)\
	$(blue)      generate ~ $(generate_bin)$(reset)$(endl)\
	$(blue)      solve    ~ $(solve_bin)$(reset)$(endl))

# ----------------------------------------------------------------- #
# Verify that the envrionment file exists and is loaded. Otherwise,
# notify the user.
.PHONY: environment
environment:
ifeq ($(environment_exists),1)
	$(call status, Loaded environment.)
else
	$(call failure, The specified make environment does not exist. \
	$(endl) $(endl)\
	$(red)$(environment_file)$(reset) \
	$(endl)$(endl)\
	Add one to the environment directory; see the README $(endl)\
	for further direction.)
endif
# ----------------------------------------------------------------- #
.PHONY: build-rules
build-rules:
ifeq ($(toolkit_makefile_exists),1)
	$(call status, Loaded build instructions.)
else
	$(call failure, The specified Makefile does not exist. \
	$(endl) $(endl)\
	$(red)$(toolkit_makefile_file)$(reset) \
	$(endl)$(endl)\
	Add one to the build directory; see the README for $(endl)\
	further direction.)
endif
# ----------------------------------------------------------------- #
.PHONY: clean
clean:
	rm -rf object target ./generate ./solve
	$(call status, Cleaned.)
# ----------------------------------------------------------------- #
