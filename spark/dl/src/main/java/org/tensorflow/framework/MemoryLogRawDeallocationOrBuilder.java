// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: log_memory.proto

package org.tensorflow.framework;

public interface MemoryLogRawDeallocationOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.MemoryLogRawDeallocation)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Process-unique step id.
   * </pre>
   *
   * <code>optional int64 step_id = 1;</code>
   */
  long getStepId();

  /**
   * <pre>
   * Name of the operation making the deallocation.
   * </pre>
   *
   * <code>optional string operation = 2;</code>
   */
  java.lang.String getOperation();
  /**
   * <pre>
   * Name of the operation making the deallocation.
   * </pre>
   *
   * <code>optional string operation = 2;</code>
   */
  com.google.protobuf.ByteString
      getOperationBytes();

  /**
   * <pre>
   * Id of the tensor buffer being deallocated, used to match to a
   * corresponding allocation.
   * </pre>
   *
   * <code>optional int64 allocation_id = 3;</code>
   */
  long getAllocationId();

  /**
   * <pre>
   * Name of the allocator used.
   * </pre>
   *
   * <code>optional string allocator_name = 4;</code>
   */
  java.lang.String getAllocatorName();
  /**
   * <pre>
   * Name of the allocator used.
   * </pre>
   *
   * <code>optional string allocator_name = 4;</code>
   */
  com.google.protobuf.ByteString
      getAllocatorNameBytes();

  /**
   * <pre>
   * True if the deallocation is queued and will be performed later,
   * e.g. for GPU lazy freeing of buffers.
   * </pre>
   *
   * <code>optional bool deferred = 5;</code>
   */
  boolean getDeferred();
}
