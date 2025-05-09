#!/usr/bin/env python
"""
Kernel stability test script - tests JAX GPU operations to verify kernel stability
"""
import os
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('kernel-test')

def test_import_paths():
    """Test that the import paths are correct and utils are accessible"""
    logger.info("Testing import paths...")
    import site
    site.addsitedir("/workspace/src")
    
    try:
        from src.utils import jax_gpu_utils
        logger.info("✅ Successfully imported jax_gpu_utils")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import jax_gpu_utils: {e}")
        return False

def test_jax_operations():
    """Test basic JAX operations"""
    logger.info("Testing JAX operations...")
    try:
        # This should trigger the jax_gpu_utils smoke test
        import jax
        import jax.numpy as jnp
        
        # Log JAX configuration
        logger.info(f"JAX version: {jax.__version__}")
        logger.info(f"JAX backend: {jax.default_backend()}")
        logger.info(f"JAX devices: {jax.devices()}")
        
        # Basic operations
        x = jnp.ones((1000, 1000))
        logger.info(f"Created array of shape {x.shape}")
        
        # Matrix multiplication (uses GPU if available)
        start = time.time()
        result = jnp.dot(x, x)
        duration = time.time() - start
        logger.info(f"Matrix multiplication completed in {duration:.4f} seconds")
        
        # Force computation to ensure GPU is used
        result.block_until_ready()
        
        # Try a larger allocation
        logger.info("Testing larger memory allocation...")
        large_x = jnp.ones((8000, 8000))
        large_result = jnp.dot(large_x, large_x)
        large_result.block_until_ready()
        logger.info("✅ Successfully completed large matrix multiplication")
        
        return True
    except Exception as e:
        logger.error(f"❌ JAX operations failed: {e}")
        return False

def main():
    logger.info("Starting kernel stability test")
    
    # Print environment variables related to GPU and JAX
    logger.info("Environment variables:")
    for var in ['JAX_PLATFORM_NAME', 'XLA_PYTHON_CLIENT_PREALLOCATE', 
                'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_ALLOCATOR']:
        logger.info(f"  {var}={os.environ.get(var, 'not set')}")
    
    # Run tests
    import_success = test_import_paths()
    if not import_success:
        logger.error("Import path test failed, aborting")
        return 1
    
    operation_success = test_jax_operations()
    if not operation_success:
        logger.error("JAX operations test failed")
        return 2
    
    logger.info("✅ All tests completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 