# Makefile for orc-ir cross-platform code
#  
#
#
# ----------------------------------------------------------------- #
# Build envrionment variables 
hostname = $(shell hostname -f)

environment_path      := ./.configure/environment
toolkit_makefile_path := build

environment_file = $(environment_path)/$(hostname) 
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


toolkit_makefile_file = $(toolkit_makefile_path)/$(toolkit).$(device) 
ifneq ("$(wildcard $(toolkit_makefile_file))","")
	toolkit_makefile_exists := 1
	# -include $(toolkit_makefile_file)
else
	toolkit_makefile_exists := 0
endif

target_directory := target
target = $(target_directory)/$(environment_slug)/$(target_file)
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

all: environment makefile

# Verify that the envrionment file exists and is loaded. Otherwise, 
# notify the user. 
#
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

.PHONY: makefile
makefile:
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
#
#
#
#