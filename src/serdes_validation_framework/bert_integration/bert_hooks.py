"""
BERT Script Hooks

This module provides a comprehensive hook system for BERT testing
with script integration and event-driven testing capabilities.
"""

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of BERT hooks"""
    PRE_TEST = auto()      # Before test starts
    POST_TEST = auto()     # After test completes
    ON_ERROR = auto()      # When error occurs
    ON_THRESHOLD = auto()  # When BER threshold exceeded
    PERIODIC = auto()      # Periodic during test
    CUSTOM = auto()        # Custom trigger


class HookTrigger(Enum):
    """Hook trigger conditions"""
    ALWAYS = auto()
    BER_THRESHOLD = auto()
    ERROR_COUNT = auto()
    TIME_INTERVAL = auto()
    SIGNAL_QUALITY = auto()
    CUSTOM_CONDITION = auto()


@dataclass
class BERTHook:
    """BERT hook definition"""
    name: str
    hook_type: HookType
    trigger: HookTrigger
    
    # Script/command to execute
    script_path: Optional[str] = None
    command: Optional[str] = None
    function: Optional[Callable] = None
    
    # Trigger conditions
    ber_threshold: float = 1e-12
    error_threshold: int = 100
    time_interval: float = 60.0  # seconds
    
    # Hook configuration
    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 3
    async_execution: bool = False
    
    # Environment variables for script
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Hook metadata
    description: str = ""
    priority: int = 100  # Lower number = higher priority
    
    def __post_init__(self):
        """Validate hook configuration"""
        if not any([self.script_path, self.command, self.function]):
            raise ValueError("Hook must have script_path, command, or function")
        
        if self.script_path and not Path(self.script_path).exists():
            logger.warning(f"Script path does not exist: {self.script_path}")


@dataclass
class HookExecutionResult:
    """Result of hook execution"""
    hook_name: str
    success: bool
    execution_time: float
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    exception: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class BERTHookManager:
    """Manager for BERT script hooks"""
    
    def __init__(self):
        """Initialize hook manager"""
        self.hooks: Dict[str, BERTHook] = {}
        self.execution_history: List[HookExecutionResult] = []
        self.active_hooks: Dict[str, threading.Thread] = {}
        
        # Hook execution statistics
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0
        }
        
        logger.info("BERT hook manager initialized")
    
    def register_hook(self, hook: BERTHook) -> bool:
        """Register a new hook"""
        try:
            if hook.name in self.hooks:
                logger.warning(f"Hook {hook.name} already exists, replacing")
            
            self.hooks[hook.name] = hook
            logger.info(f"Hook registered: {hook.name} ({hook.hook_type.name})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register hook {hook.name}: {e}")
            return False
    
    def unregister_hook(self, hook_name: str) -> bool:
        """Unregister a hook"""
        try:
            if hook_name in self.hooks:
                del self.hooks[hook_name]
                logger.info(f"Hook unregistered: {hook_name}")
                return True
            else:
                logger.warning(f"Hook not found: {hook_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister hook {hook_name}: {e}")
            return False
    
    def enable_hook(self, hook_name: str) -> bool:
        """Enable a hook"""
        if hook_name in self.hooks:
            self.hooks[hook_name].enabled = True
            logger.info(f"Hook enabled: {hook_name}")
            return True
        return False
    
    def disable_hook(self, hook_name: str) -> bool:
        """Disable a hook"""
        if hook_name in self.hooks:
            self.hooks[hook_name].enabled = False
            logger.info(f"Hook disabled: {hook_name}")
            return True
        return False
    
    def execute_hooks(self, 
                     hook_type: HookType,
                     context: Dict[str, Any]) -> List[HookExecutionResult]:
        """Execute all hooks of specified type"""
        results = []
        
        # Get hooks of specified type, sorted by priority
        applicable_hooks = [
            hook for hook in self.hooks.values()
            if hook.hook_type == hook_type and hook.enabled
        ]
        applicable_hooks.sort(key=lambda h: h.priority)
        
        for hook in applicable_hooks:
            if self._should_trigger_hook(hook, context):
                result = self._execute_hook(hook, context)
                results.append(result)
                self.execution_history.append(result)
                
                # Update statistics
                self._update_execution_stats(result)
        
        return results
    
    def execute_hook_by_name(self, 
                           hook_name: str,
                           context: Dict[str, Any]) -> Optional[HookExecutionResult]:
        """Execute specific hook by name"""
        if hook_name not in self.hooks:
            logger.error(f"Hook not found: {hook_name}")
            return None
        
        hook = self.hooks[hook_name]
        if not hook.enabled:
            logger.warning(f"Hook disabled: {hook_name}")
            return None
        
        result = self._execute_hook(hook, context)
        self.execution_history.append(result)
        self._update_execution_stats(result)
        
        return result
    
    def get_hook_status(self) -> Dict[str, Any]:
        """Get status of all hooks"""
        status = {
            'total_hooks': len(self.hooks),
            'enabled_hooks': sum(1 for h in self.hooks.values() if h.enabled),
            'disabled_hooks': sum(1 for h in self.hooks.values() if not h.enabled),
            'active_executions': len(self.active_hooks),
            'execution_stats': self.execution_stats,
            'hooks': {}
        }
        
        for name, hook in self.hooks.items():
            status['hooks'][name] = {
                'type': hook.hook_type.name,
                'trigger': hook.trigger.name,
                'enabled': hook.enabled,
                'priority': hook.priority,
                'description': hook.description
            }
        
        return status
    
    def get_execution_history(self, count: int = 100) -> List[HookExecutionResult]:
        """Get recent execution history"""
        return self.execution_history[-count:] if len(self.execution_history) >= count else self.execution_history
    
    def clear_execution_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        logger.info("Execution history cleared")
    
    def export_hooks_config(self, filename: str) -> bool:
        """Export hooks configuration to file"""
        try:
            config = {
                'hooks': {},
                'export_timestamp': time.time()
            }
            
            for name, hook in self.hooks.items():
                config['hooks'][name] = {
                    'name': hook.name,
                    'hook_type': hook.hook_type.name,
                    'trigger': hook.trigger.name,
                    'script_path': hook.script_path,
                    'command': hook.command,
                    'ber_threshold': hook.ber_threshold,
                    'error_threshold': hook.error_threshold,
                    'time_interval': hook.time_interval,
                    'enabled': hook.enabled,
                    'timeout_seconds': hook.timeout_seconds,
                    'retry_count': hook.retry_count,
                    'async_execution': hook.async_execution,
                    'environment': hook.environment,
                    'description': hook.description,
                    'priority': hook.priority
                }
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Hooks configuration exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export hooks config: {e}")
            return False
    
    def import_hooks_config(self, filename: str) -> bool:
        """Import hooks configuration from file"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            for _name, hook_config in config.get('hooks', {}).items():
                hook = BERTHook(
                    name=hook_config['name'],
                    hook_type=HookType[hook_config['hook_type']],
                    trigger=HookTrigger[hook_config['trigger']],
                    script_path=hook_config.get('script_path'),
                    command=hook_config.get('command'),
                    ber_threshold=hook_config.get('ber_threshold', 1e-12),
                    error_threshold=hook_config.get('error_threshold', 100),
                    time_interval=hook_config.get('time_interval', 60.0),
                    enabled=hook_config.get('enabled', True),
                    timeout_seconds=hook_config.get('timeout_seconds', 30.0),
                    retry_count=hook_config.get('retry_count', 3),
                    async_execution=hook_config.get('async_execution', False),
                    environment=hook_config.get('environment', {}),
                    description=hook_config.get('description', ''),
                    priority=hook_config.get('priority', 100)
                )
                
                self.register_hook(hook)
            
            logger.info(f"Hooks configuration imported from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import hooks config: {e}")
            return False
    
    def _should_trigger_hook(self, hook: BERTHook, context: Dict[str, Any]) -> bool:
        """Check if hook should be triggered based on context"""
        try:
            if hook.trigger == HookTrigger.ALWAYS:
                return True
            
            elif hook.trigger == HookTrigger.BER_THRESHOLD:
                current_ber = context.get('ber', 0.0)
                return current_ber > hook.ber_threshold
            
            elif hook.trigger == HookTrigger.ERROR_COUNT:
                error_count = context.get('error_count', 0)
                return error_count > hook.error_threshold
            
            elif hook.trigger == HookTrigger.TIME_INTERVAL:
                last_execution = context.get('last_hook_execution', {}).get(hook.name, 0)
                return (time.time() - last_execution) >= hook.time_interval
            
            elif hook.trigger == HookTrigger.SIGNAL_QUALITY:
                signal_quality = context.get('signal_quality', 1.0)
                return signal_quality < 0.8  # Threshold for poor signal quality
            
            elif hook.trigger == HookTrigger.CUSTOM_CONDITION:
                # Custom condition function can be provided in context
                custom_check = context.get('custom_condition_check')
                if custom_check and callable(custom_check):
                    return custom_check(hook, context)
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking hook trigger for {hook.name}: {e}")
            return False
    
    def _execute_hook(self, hook: BERTHook, context: Dict[str, Any]) -> HookExecutionResult:
        """Execute a single hook"""
        start_time = time.time()
        
        try:
            logger.info(f"Executing hook: {hook.name}")
            
            if hook.async_execution:
                # Execute asynchronously
                thread = threading.Thread(
                    target=self._execute_hook_sync,
                    args=(hook, context)
                )
                thread.daemon = True
                thread.start()
                self.active_hooks[hook.name] = thread
                
                return HookExecutionResult(
                    hook_name=hook.name,
                    success=True,
                    execution_time=time.time() - start_time,
                    stdout="Async execution started"
                )
            else:
                # Execute synchronously
                return self._execute_hook_sync(hook, context)
                
        except Exception as e:
            logger.error(f"Hook execution failed: {hook.name}: {e}")
            return HookExecutionResult(
                hook_name=hook.name,
                success=False,
                execution_time=time.time() - start_time,
                exception=str(e)
            )
    
    def _execute_hook_sync(self, hook: BERTHook, context: Dict[str, Any]) -> HookExecutionResult:
        """Execute hook synchronously"""
        start_time = time.time()
        
        try:
            # Prepare environment
            env = dict(os.environ) if 'os' in globals() else {}
            env.update(hook.environment)
            
            # Add context variables to environment
            env.update({
                f"BERT_{k.upper()}": str(v) for k, v in context.items()
                if isinstance(v, (str, int, float, bool))
            })
            
            if hook.function:
                # Execute Python function
                result = hook.function(context)
                return HookExecutionResult(
                    hook_name=hook.name,
                    success=True,
                    execution_time=time.time() - start_time,
                    stdout=str(result) if result else ""
                )
            
            elif hook.script_path:
                # Execute script file
                cmd = [str(hook.script_path)]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=hook.timeout_seconds,
                    env=env
                )
                
                return HookExecutionResult(
                    hook_name=hook.name,
                    success=result.returncode == 0,
                    execution_time=time.time() - start_time,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr
                )
            
            elif hook.command:
                # Execute shell command
                result = subprocess.run(
                    hook.command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=hook.timeout_seconds,
                    env=env
                )
                
                return HookExecutionResult(
                    hook_name=hook.name,
                    success=result.returncode == 0,
                    execution_time=time.time() - start_time,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr
                )
            
            else:
                raise ValueError("No execution method specified")
                
        except subprocess.TimeoutExpired:
            return HookExecutionResult(
                hook_name=hook.name,
                success=False,
                execution_time=time.time() - start_time,
                exception="Execution timeout"
            )
        
        except Exception as e:
            return HookExecutionResult(
                hook_name=hook.name,
                success=False,
                execution_time=time.time() - start_time,
                exception=str(e)
            )
        
        finally:
            # Clean up async execution tracking
            if hook.name in self.active_hooks:
                del self.active_hooks[hook.name]
    
    def _update_execution_stats(self, result: HookExecutionResult):
        """Update execution statistics"""
        self.execution_stats['total_executions'] += 1
        
        if result.success:
            self.execution_stats['successful_executions'] += 1
        else:
            self.execution_stats['failed_executions'] += 1
        
        # Update average execution time
        total = self.execution_stats['total_executions']
        current_avg = self.execution_stats['average_execution_time']
        new_avg = ((current_avg * (total - 1)) + result.execution_time) / total
        self.execution_stats['average_execution_time'] = new_avg


def create_bert_hook_manager() -> BERTHookManager:
    """Factory function to create BERT hook manager"""
    return BERTHookManager()


def create_standard_hooks() -> List[BERTHook]:
    """Create standard BERT hooks"""
    hooks = []
    
    # Pre-test setup hook
    hooks.append(BERTHook(
        name="pre_test_setup",
        hook_type=HookType.PRE_TEST,
        trigger=HookTrigger.ALWAYS,
        command="echo 'Starting BERT test at $(date)'",
        description="Log test start time",
        priority=10
    ))
    
    # Post-test cleanup hook
    hooks.append(BERTHook(
        name="post_test_cleanup",
        hook_type=HookType.POST_TEST,
        trigger=HookTrigger.ALWAYS,
        command="echo 'BERT test completed at $(date)'",
        description="Log test completion time",
        priority=90
    ))
    
    # BER threshold alert hook
    hooks.append(BERTHook(
        name="ber_threshold_alert",
        hook_type=HookType.ON_THRESHOLD,
        trigger=HookTrigger.BER_THRESHOLD,
        command="echo 'BER threshold exceeded: $BERT_BER'",
        ber_threshold=1e-9,
        description="Alert when BER exceeds threshold",
        priority=20
    ))
    
    # Periodic status hook
    hooks.append(BERTHook(
        name="periodic_status",
        hook_type=HookType.PERIODIC,
        trigger=HookTrigger.TIME_INTERVAL,
        command="echo 'BERT status: BER=$BERT_BER, Errors=$BERT_ERROR_COUNT'",
        time_interval=300.0,  # Every 5 minutes
        description="Periodic status logging",
        priority=50
    ))
    
    return hooks


# Import os for environment variables
import os

# Export main classes
__all__ = [
    'BERTHookManager',
    'BERTHook',
    'HookType',
    'HookTrigger',
    'HookExecutionResult',
    'create_bert_hook_manager',
    'create_standard_hooks'
]
