# Makefile for orc-ir cross-platform code
#  
#
#
# ----------------------------------------------------------------- #
# Build envrionment variables 
hostname = $(shell hostname -f)
environment_path = ./.configure/environment
environment_file = $(environment_path)/$(hostname) 
ifneq ("$(wildcard $(environment_file))","") 
    environment_exists := 1
	-include $(environment_file) 
else
    environment_exists := 0
endif
# ----------------------------------------------------------------- #
# Pretty printing and error reporting. 
esc   := $(shell printf '\033')             # Ascii color variables #
blue  := $(esc)[34m.            
red   := $(esc)[31m
reset := $(esc)[0m
define endl                                      # Newline variable #


endef
define status                                     # Status function #
$(info $(blue)(make)$(reset)$(1))  
endef
define failure                                    # error funtction #
$(info $(red)(error)$(reset)$(1))
@exit 2
endef
# ----------------------------------------------------------------- #

all: environment


# Verify that the envrionment file exists and is loaded. Otherwise, 
# notify the user. 
#
.PHONY: environment
environment:
ifeq ($(environment_exists),1)
	$(call status, Loaded environment)
else
	$(call failure, The specified make environment does not exist. \
	$(endl) $(endl)\
	$(red)$(environment_file)$(reset) \
	$(endl)$(endl)\
	Add one to the environment directory; see the README $(endl)\
	for further direction.)
endif