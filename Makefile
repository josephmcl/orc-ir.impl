


hostname = $(shell hostname -f)
environment_path = ./.configure/environment
environment_file = $(environment_path)/$(hostname)

ifneq ("$(wildcard $(environment_file))","")
    environment_exists := 1
	-include $(environment_file)
else
    environment_exists := 0
endif

esc   := $(shell printf '\033')
blue  := $(esc)[34m
red   := $(esc)[31m
reset := $(esc)[0m

define status
$(info $(blue)(make)$(reset)$(1))
endef

define failure
$(info $(red)(error)$(reset)$(1))
@exit 2
endef

all: environment


.PHONY: environment
environment:
ifeq ($(environment_exists),1)
	$(call status, Loaded environment $(blue)$(environment_file)$(reset))
else
	$(call failure, The specified make environment $(red)$(environment_file)"\
	$(reset)does not exist. Add one to the environments directory \
	($(red)$(environment_path)$(reset)). See the README for further \
	direction.)
endif