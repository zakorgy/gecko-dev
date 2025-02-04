/*
 * Copyright 2017 WebAssembly Community Group participants
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

const kC0DEFEFE = new Uint8Array([0xC0, 0xDE, 0xFE, 0xFE]);

(function testJSAPI() {

const WasmPage = 64 * 1024;

const emptyModuleBinary = new WasmModuleBuilder().toBuffer();

const importingModuleBinary = (() => {
    let builder = new WasmModuleBuilder();

    builder.addImport('', 'f', kSig_v_v);

    return builder.toBuffer();
})();

const complexImportingModuleBinary = (() => {
    let builder = new WasmModuleBuilder();

    builder.addImport('a', 'b', kSig_v_v);
    builder.addImportedMemory('c', 'd', 1);
    builder.addImportedTable('e', 'f', 1);
    builder.addImportedGlobal('g', '⚡', kWasmI32);

    return builder.toBuffer();
})();

const exportingModuleBinary = (() => {
    let builder = new WasmModuleBuilder();

    builder
        .addFunction('f', kSig_i_v)
        .addBody([
            kExprI32Const,
            42,
            kExprEnd
        ])
        .exportFunc();

    return builder.toBuffer();
})();

const complexExportingModuleBinary = (() => {
    let builder = new WasmModuleBuilder();

    builder
        .addFunction('a', kSig_v_v)
        .addBody([
            kExprEnd
        ])
        .exportFunc();

    builder.addMemory(1, 1, /* exported */ false);
    builder.exportMemoryAs('b');

    builder.setFunctionTableLength(1);
    builder.addExportOfKind('c', kExternalTable, 0);

    // Default init for global values is 0. Keep that.
    builder.addGlobal(kWasmI32, /* mutable */ false)
        .exportAs("⚡");

    return builder.toBuffer();
})();

const moduleBinaryImporting2Memories = (() => {
    var builder = new WasmModuleBuilder();
    builder.addImportedMemory("", "memory1");
    builder.addImportedMemory("", "memory2");
    return builder.toBuffer();
})();

const moduleBinaryWithMemSectionAndMemImport = (() => {
    var builder = new WasmModuleBuilder();
    builder.addMemory(1, 1, false);
    builder.addImportedMemory("", "memory1");
    return builder.toBuffer();
})();

const exportingModuleIdentityFn = (() => {
    let builder = new WasmModuleBuilder();

    builder
        .addFunction('id', kSig_i_i)
        .addBody([kExprGetLocal, 0, kExprEnd])
        .exportFunc();

    return builder.toBuffer();
})();

let Module;
let Instance;
let CompileError;
let LinkError;
let RuntimeError;
let Memory;
let instanceProto;
let memoryProto;
let mem1;
let Table;
let tbl1;
let tableProto;
let Global;
let globalProto;
let globalI32;
let globalI64;
let globalF32;
let globalF64;
let globalI32Mut;
let globalI64Mut;
let globalF32Mut;
let globalF64Mut;

let emptyModule;
let exportingModule;
let exportingInstance;
let exportsObj;
let importingModule;

// Start of tests.

test(() => {
    const wasmDesc = Object.getOwnPropertyDescriptor(this, 'WebAssembly');
    assert_equals(typeof wasmDesc.value, "object");
    assert_true(wasmDesc.writable);
    assert_false(wasmDesc.enumerable);
    assert_true(wasmDesc.configurable);
}, "'WebAssembly' data property on global object");

test(() => {
    const wasmDesc = Object.getOwnPropertyDescriptor(this, 'WebAssembly');
    assert_equals(WebAssembly, wasmDesc.value);
    assert_equals(String(WebAssembly), "[object WebAssembly]");
}, "'WebAssembly' object");

test(() => {
    const compileErrorDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'CompileError');
    const linkErrorDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'LinkError');
    const runtimeErrorDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'RuntimeError');
    assert_equals(typeof compileErrorDesc.value, "function");
    assert_equals(typeof linkErrorDesc.value, "function");
    assert_equals(typeof runtimeErrorDesc.value, "function");
    assert_equals(compileErrorDesc.writable, true);
    assert_equals(linkErrorDesc.writable, true);
    assert_equals(runtimeErrorDesc.writable, true);
    assert_equals(compileErrorDesc.enumerable, false);
    assert_equals(linkErrorDesc.enumerable, false);
    assert_equals(runtimeErrorDesc.enumerable, false);
    assert_equals(compileErrorDesc.configurable, true);
    assert_equals(linkErrorDesc.configurable, true);
    assert_equals(runtimeErrorDesc.configurable, true);

    CompileError = WebAssembly.CompileError;
    LinkError = WebAssembly.LinkError;
    RuntimeError = WebAssembly.RuntimeError;
}, "'WebAssembly.(Compile|Link|Runtime)Error' data property");

test(() => {
    const compileErrorDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'CompileError');
    const linkErrorDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'LinkError');
    const runtimeErrorDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'RuntimeError');
    assert_equals(CompileError, compileErrorDesc.value);
    assert_equals(LinkError, linkErrorDesc.value);
    assert_equals(RuntimeError, runtimeErrorDesc.value);
    assert_equals(CompileError.length, 1);
    assert_equals(LinkError.length, 1);
    assert_equals(RuntimeError.length, 1);
    assert_equals(CompileError.name, "CompileError");
    assert_equals(LinkError.name, "LinkError");
    assert_equals(RuntimeError.name, "RuntimeError");
}, "'WebAssembly.(Compile|Runtime)Error' constructor function");

test(() => {
    const compileError = new CompileError;
    const runtimeError = new RuntimeError;
    assert_equals(compileError instanceof CompileError, true);
    assert_equals(runtimeError instanceof RuntimeError, true);
    assert_equals(compileError instanceof Error, true);
    assert_equals(runtimeError instanceof Error, true);
    assert_equals(compileError instanceof TypeError, false);
    assert_equals(runtimeError instanceof TypeError, false);
    assert_equals(compileError.message, "");
    assert_equals(runtimeError.message, "");
    assert_equals(new CompileError("hi").message, "hi");
    assert_equals(new RuntimeError("hi").message, "hi");
}, "'WebAssembly.(Compile|Runtime)Error' instance objects");

test(() => {
    const moduleDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Module');
    assert_equals(typeof moduleDesc.value, "function");
    assert_equals(moduleDesc.writable, true);
    assert_equals(moduleDesc.enumerable, false);
    assert_equals(moduleDesc.configurable, true);
    Module = WebAssembly.Module;
}, "'WebAssembly.Module' data property");

test(() => {
    const moduleDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Module');
    assert_equals(Module, moduleDesc.value);
    assert_equals(Module.length, 1);
    assert_equals(Module.name, "Module");
    assertThrows(() => Module(), TypeError);
    assertThrows(() => new Module(), TypeError);
    assertThrows(() => new Module(undefined), TypeError);
    assertThrows(() => new Module(1), TypeError);
    assertThrows(() => new Module({}), TypeError);
    assertThrows(() => new Module(new Uint8Array()), CompileError);
    assertThrows(() => new Module(new ArrayBuffer()), CompileError);
    assert_equals(new Module(emptyModuleBinary) instanceof Module, true);
    assert_equals(new Module(new Uint8Array(emptyModuleBinary)) instanceof Module, true);
}, "'WebAssembly.Module' constructor function");

test(() => {
    const moduleProtoDesc = Object.getOwnPropertyDescriptor(Module, 'prototype');
    assert_equals(typeof moduleProtoDesc.value, "object");
    assert_equals(moduleProtoDesc.writable, false);
    assert_equals(moduleProtoDesc.enumerable, false);
    assert_equals(moduleProtoDesc.configurable, false);
}, "'WebAssembly.Module.prototype' data property");

test(() => {
    const moduleProtoDesc = Object.getOwnPropertyDescriptor(Module, 'prototype');
    const moduleProto = Module.prototype;
    assert_equals(moduleProto, moduleProtoDesc.value);
    assert_equals(String(moduleProto), "[object WebAssembly.Module]");
    assert_equals(Object.getPrototypeOf(moduleProto), Object.prototype);
}, "'WebAssembly.Module.prototype' object");

test(() => {
    const moduleProto = Module.prototype;
    emptyModule = new Module(emptyModuleBinary);
    exportingModule = new Module(exportingModuleBinary);
    importingModule = new Module(importingModuleBinary);
    assert_equals(typeof emptyModule, "object");
    assert_equals(String(emptyModule), "[object WebAssembly.Module]");
    assert_equals(Object.getPrototypeOf(emptyModule), moduleProto);
}, "'WebAssembly.Module' instance objects");

test(() => {
    const moduleImportsDesc = Object.getOwnPropertyDescriptor(Module, 'imports');
    assert_equals(typeof moduleImportsDesc.value, "function");
    assert_equals(moduleImportsDesc.writable, true);
    assert_equals(moduleImportsDesc.enumerable, true);
    assert_equals(moduleImportsDesc.configurable, true);
}, "'WebAssembly.Module.imports' data property");

test(() => {
    const moduleImportsDesc = Object.getOwnPropertyDescriptor(Module, 'imports');
    const moduleImports = moduleImportsDesc.value;
    assert_equals(moduleImports.length, 1);
    assertThrows(() => moduleImports(), TypeError);
    assertThrows(() => moduleImports(undefined), TypeError);
    assertThrows(() => moduleImports({}), TypeError);
    var arr = moduleImports(emptyModule);
    assert_equals(arr instanceof Array, true);
    assert_equals(arr.length, 0);
    var arr = moduleImports(new Module(complexImportingModuleBinary));
    assert_equals(arr instanceof Array, true);
    assert_equals(arr.length, 4);
    assert_equals(arr[0].kind, "function");
    assert_equals(arr[0].module, "a");
    assert_equals(arr[0].name, "b");
    assert_equals(arr[1].kind, "memory");
    assert_equals(arr[1].module, "c");
    assert_equals(arr[1].name, "d");
    assert_equals(arr[2].kind, "table");
    assert_equals(arr[2].module, "e");
    assert_equals(arr[2].name, "f");
    assert_equals(arr[3].kind, "global");
    assert_equals(arr[3].module, "g");
    assert_equals(arr[3].name, "⚡");
}, "'WebAssembly.Module.imports' method");

test(() => {
    const moduleExportsDesc = Object.getOwnPropertyDescriptor(Module, 'exports');
    assert_equals(typeof moduleExportsDesc.value, "function");
    assert_equals(moduleExportsDesc.writable, true);
    assert_equals(moduleExportsDesc.enumerable, true);
    assert_equals(moduleExportsDesc.configurable, true);
}, "'WebAssembly.Module.exports' data property");

test(() => {
    const moduleExportsDesc = Object.getOwnPropertyDescriptor(Module, 'exports');
    const moduleExports = moduleExportsDesc.value;
    assert_equals(moduleExports.length, 1);
    assertThrows(() => moduleExports(), TypeError);
    assertThrows(() => moduleExports(undefined), TypeError);
    assertThrows(() => moduleExports({}), TypeError);
    var arr = moduleExports(emptyModule);
    assert_equals(arr instanceof Array, true);
    assert_equals(arr.length, 0);
    var arr = moduleExports(new Module(complexExportingModuleBinary));
    assert_equals(arr instanceof Array, true);
    assert_equals(arr.length, 4);
    assert_equals(arr[0].kind, "function");
    assert_equals(arr[0].name, "a");
    assert_equals(arr[1].kind, "memory");
    assert_equals(arr[1].name, "b");
    assert_equals(arr[2].kind, "table");
    assert_equals(arr[2].name, "c");
    assert_equals(arr[3].kind, "global");
    assert_equals(arr[3].name, "⚡");
}, "'WebAssembly.Module.exports' method");

test(() => {
    const customSectionsDesc = Object.getOwnPropertyDescriptor(Module, 'customSections');
    assert_equals(typeof customSectionsDesc.value, "function");
    assert_equals(customSectionsDesc.writable, true);
    assert_equals(customSectionsDesc.enumerable, true);
    assert_equals(customSectionsDesc.configurable, true);
}, "'WebAssembly.Module.customSections' data property");

test(() => {
    const customSectionsDesc = Object.getOwnPropertyDescriptor(Module, 'customSections');
    const moduleCustomSections = customSectionsDesc.value;
    assert_equals(moduleCustomSections.length, 2);
    assertThrows(() => moduleCustomSections(), TypeError);
    assertThrows(() => moduleCustomSections(undefined), TypeError);
    assertThrows(() => moduleCustomSections({}), TypeError);
    var arr = moduleCustomSections(emptyModule, "abracadabra");
    assert_equals(arr instanceof Array, true);
    assert_equals(arr.length, 0);
}, "'WebAssembly.Module.customSections' method");

test(() => {
    const instanceDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Instance');
    assert_equals(typeof instanceDesc.value, "function");
    assert_equals(instanceDesc.writable, true);
    assert_equals(instanceDesc.enumerable, false);
    assert_equals(instanceDesc.configurable, true);
    Instance = WebAssembly.Instance;
}, "'WebAssembly.Instance' data property");

test(() => {
    const instanceDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Instance');
    assert_equals(Instance, instanceDesc.value);
    assert_equals(Instance.length, 1);
    assert_equals(Instance.name, "Instance");
    assertThrows(() => Instance(), TypeError);
    assertThrows(() => new Instance(1), TypeError);
    assertThrows(() => new Instance({}), TypeError);
    assertThrows(() => new Instance(emptyModule, null), TypeError);
    assertThrows(() => new Instance(importingModule, null), TypeError);
    assertThrows(() => new Instance(importingModule, undefined), TypeError);
    assertThrows(() => new Instance(importingModule, {}), TypeError);
    assertThrows(() => new Instance(importingModule, {"":{g:()=>{}}}), LinkError);
    assertThrows(() => new Instance(importingModule, {t:{f:()=>{}}}), TypeError);
    assert_equals(new Instance(emptyModule) instanceof Instance, true);
    assert_equals(new Instance(emptyModule, {}) instanceof Instance, true);
}, "'WebAssembly.Instance' constructor function");

test(() => {
    const instanceProtoDesc = Object.getOwnPropertyDescriptor(Instance, 'prototype');
    assert_equals(typeof instanceProtoDesc.value, "object");
    assert_equals(instanceProtoDesc.writable, false);
    assert_equals(instanceProtoDesc.enumerable, false);
    assert_equals(instanceProtoDesc.configurable, false);
}, "'WebAssembly.Instance.prototype' data property");

test(() => {
    instanceProto = Instance.prototype;
    const instanceProtoDesc = Object.getOwnPropertyDescriptor(Instance, 'prototype');
    assert_equals(instanceProto, instanceProtoDesc.value);
    assert_equals(String(instanceProto), "[object WebAssembly.Instance]");
    assert_equals(Object.getPrototypeOf(instanceProto), Object.prototype);
}, "'WebAssembly.Instance.prototype' object");

test(() => {
    const instanceProto = Instance.prototype;
    exportingInstance = new Instance(exportingModule);
    assert_equals(typeof exportingInstance, "object");
    assert_equals(String(exportingInstance), "[object WebAssembly.Instance]");
    assert_equals(Object.getPrototypeOf(exportingInstance), instanceProto);
}, "'WebAssembly.Instance' instance objects");

test(() => {
    const exportsDesc = Object.getOwnPropertyDescriptor(instanceProto, 'exports');
    assert_equals(typeof exportsDesc.get, "function");
    assert_equals(exportsDesc.set, undefined);
    assert_equals(exportsDesc.enumerable, true);
    assert_equals(exportsDesc.configurable, true);
    const exportsGetter = exportsDesc.get;
    assertThrows(() => exportsGetter.call(), TypeError);
    assertThrows(() => exportsGetter.call({}), TypeError);
    assert_equals(typeof exportsGetter.call(exportingInstance), "object");
}, "'WebAssembly.Instance.prototype.exports' accessor property");

test(() => {
    exportsObj = exportingInstance.exports;
    assert_equals(typeof exportsObj, "object");
    assert_equals(Object.isExtensible(exportsObj), false);
    assert_equals(Object.getPrototypeOf(exportsObj), null);
    assert_equals(Object.keys(exportsObj).join(), "f");
    exportsObj.g = 1;
    assert_equals(Object.keys(exportsObj).join(), "f");
    assertThrows(() => Object.setPrototypeOf(exportsObj, {}), TypeError);
    assert_equals(Object.getPrototypeOf(exportsObj), null);
    assertThrows(() => Object.defineProperty(exportsObj, 'g', {}), TypeError);
    assert_equals(Object.keys(exportsObj).join(), "f");
}, "exports object");

test(() => {
    const f = exportsObj.f;
    assert_equals(f instanceof Function, true);
    assert_equals(f.length, 0);
    assert_equals('name' in f, true);
    assert_equals(Function.prototype.call.call(f), 42);
    assert_equals('prototype' in f, false);
    assertThrows(() => new f(), TypeError);
}, "Exported WebAssembly functions");

test(() => {
    const memoryDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Memory');
    assert_equals(typeof memoryDesc.value, "function");
    assert_equals(memoryDesc.writable, true);
    assert_equals(memoryDesc.enumerable, false);
    assert_equals(memoryDesc.configurable, true);
    Memory = WebAssembly.Memory;
}, "'WebAssembly.Memory' data property");

test(() => {
    const memoryDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Memory');
    assert_equals(Memory, memoryDesc.value);
    assert_equals(Memory.length, 1);
    assert_equals(Memory.name, "Memory");
    assertThrows(() => Memory(), TypeError);
    assertThrows(() => new Memory(1), TypeError);
    assertThrows(() => new Memory({initial:{valueOf() { throw new Error("here")}}}), Error);
    assertThrows(() => new Memory({initial:-1}), TypeError);
    assertThrows(() => new Memory({initial:Math.pow(2,32)}), TypeError);
    assertThrows(() => new Memory({initial:1, maximum: Math.pow(2,32)/Math.pow(2,14) }), RangeError);
    assertThrows(() => new Memory({initial:2, maximum:1 }), RangeError);
    assertThrows(() => new Memory({maximum: -1 }), TypeError);
    assert_equals(new Memory({initial:1}) instanceof Memory, true);
    assert_equals(new Memory({initial:1.5}).buffer.byteLength, WasmPage);
}, "'WebAssembly.Memory' constructor function");

test(() => {
    const memoryProtoDesc = Object.getOwnPropertyDescriptor(Memory, 'prototype');
    assert_equals(typeof memoryProtoDesc.value, "object");
    assert_equals(memoryProtoDesc.writable, false);
    assert_equals(memoryProtoDesc.enumerable, false);
    assert_equals(memoryProtoDesc.configurable, false);
}, "'WebAssembly.Memory.prototype' data property");

test(() => {
    memoryProto = Memory.prototype;
    const memoryProtoDesc = Object.getOwnPropertyDescriptor(Memory, 'prototype');
    assert_equals(memoryProto, memoryProtoDesc.value);
    assert_equals(String(memoryProto), "[object WebAssembly.Memory]");
    assert_equals(Object.getPrototypeOf(memoryProto), Object.prototype);
}, "'WebAssembly.Memory.prototype' object");

test(() => {
    mem1 = new Memory({initial:1});
    assert_equals(typeof mem1, "object");
    assert_equals(String(mem1), "[object WebAssembly.Memory]");
    assert_equals(Object.getPrototypeOf(mem1), memoryProto);
}, "'WebAssembly.Memory' instance objects");

test(() => {
    const bufferDesc = Object.getOwnPropertyDescriptor(memoryProto, 'buffer');
    assert_equals(typeof bufferDesc.get, "function");
    assert_equals(bufferDesc.set, undefined);
    assert_equals(bufferDesc.enumerable, true);
    assert_equals(bufferDesc.configurable, true);
}, "'WebAssembly.Memory.prototype.buffer' accessor property");

test(() => {
    const bufferDesc = Object.getOwnPropertyDescriptor(memoryProto, 'buffer');
    const bufferGetter = bufferDesc.get;
    assertThrows(() => bufferGetter.call(), TypeError);
    assertThrows(() => bufferGetter.call({}), TypeError);
    assert_equals(bufferGetter.call(mem1) instanceof ArrayBuffer, true);
    assert_equals(bufferGetter.call(mem1).byteLength, WasmPage);
}, "'WebAssembly.Memory.prototype.buffer' getter");

test(() => {
    const memGrowDesc = Object.getOwnPropertyDescriptor(memoryProto, 'grow');
    assert_equals(typeof memGrowDesc.value, "function");
    assert_equals(memGrowDesc.enumerable, true);
    assert_equals(memGrowDesc.configurable, true);
}, "'WebAssembly.Memory.prototype.grow' data property");

test(() => {
    const memGrowDesc = Object.getOwnPropertyDescriptor(memoryProto, 'grow');
    const memGrow = memGrowDesc.value;
    assert_equals(memGrow.length, 1);
    assertThrows(() => memGrow.call(), TypeError);
    assertThrows(() => memGrow.call({}), TypeError);
    assertThrows(() => memGrow.call(mem1, -1), TypeError);
    assertThrows(() => memGrow.call(mem1, Math.pow(2,32)), TypeError);
    var mem = new Memory({initial:1, maximum:2});
    var buf = mem.buffer;
    assert_equals(buf.byteLength, WasmPage);
    assert_equals(mem.grow(0), 1);
    assert_not_equals(buf, mem.buffer)
    assert_equals(buf.byteLength, 0);
    buf = mem.buffer;
    assert_equals(buf.byteLength, WasmPage);
    assert_equals(mem.grow(1), 1);
    assert_not_equals(buf, mem.buffer)
    assert_equals(buf.byteLength, 0);
    buf = mem.buffer;
    assert_equals(buf.byteLength, 2 * WasmPage);
    assertThrows(() => mem.grow(1), Error);
    assert_equals(buf, mem.buffer);
    mem = new Memory({initial:0, maximum:1});
    buf = mem.buffer;
    assert_equals(buf.byteLength, 0);
    assert_equals(mem.grow(0), 0);
    assert_not_equals(buf, mem.buffer)
    assert_equals(buf.byteLength, 0);
    assert_equals(mem.buffer.byteLength, 0);
}, "'WebAssembly.Memory.prototype.grow' method");

test(() => {
    const tableDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Table');
    assert_equals(typeof tableDesc.value, "function");
    assert_equals(tableDesc.writable, true);
    assert_equals(tableDesc.enumerable, false);
    assert_equals(tableDesc.configurable, true);
    Table = WebAssembly.Table;
}, "'WebAssembly.Table' data property");

test(() => {
    const tableDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Table');
    assert_equals(Table, tableDesc.value);
    assert_equals(Table.length, 1);
    assert_equals(Table.name, "Table");
    assertThrows(() => Table(), TypeError);
    assertThrows(() => new Table(1), TypeError);
    assertThrows(() => new Table({initial:1, element:1}), TypeError);
    assertThrows(() => new Table({initial:1, element:"any"}), TypeError);
    assertThrows(() => new Table({initial:1, element:{valueOf() { return "anyfunc" }}}), TypeError);
    assertThrows(() => new Table({initial:{valueOf() { throw new Error("here")}}, element:"anyfunc"}), Error);
    assertThrows(() => new Table({initial:-1, element:"anyfunc"}), TypeError);
    assertThrows(() => new Table({initial:Math.pow(2,32), element:"anyfunc"}), TypeError);
    assertThrows(() => new Table({initial:2, maximum:1, element:"anyfunc"}), RangeError);
    assertThrows(() => new Table({initial:2, maximum:Math.pow(2,32), element:"anyfunc"}), TypeError);
    assert_equals(new Table({initial:1, element:"anyfunc"}) instanceof Table, true);
    assert_equals(new Table({initial:1.5, element:"anyfunc"}) instanceof Table, true);
    assert_equals(new Table({initial:1, maximum:1.5, element:"anyfunc"}) instanceof Table, true);
    assertThrows(() => new Table({initial:1, maximum:Math.pow(2,32)-1, element:"anyfunc"}), RangeError);
}, "'WebAssembly.Table' constructor function");

test(() => {
    const tableProtoDesc = Object.getOwnPropertyDescriptor(Table, 'prototype');
    assert_equals(typeof tableProtoDesc.value, "object");
    assert_equals(tableProtoDesc.writable, false);
    assert_equals(tableProtoDesc.enumerable, false);
    assert_equals(tableProtoDesc.configurable, false);
}, "'WebAssembly.Table.prototype' data property");

test(() => {
    const tableProtoDesc = Object.getOwnPropertyDescriptor(Table, 'prototype');
    tableProto = Table.prototype;
    assert_equals(tableProto, tableProtoDesc.value);
    assert_equals(String(tableProto), "[object WebAssembly.Table]");
    assert_equals(Object.getPrototypeOf(tableProto), Object.prototype);
}, "'WebAssembly.Table.prototype' object");

test(() => {
    tbl1 = new Table({initial:2, element:"anyfunc"});
    assert_equals(typeof tbl1, "object");
    assert_equals(String(tbl1), "[object WebAssembly.Table]");
    assert_equals(Object.getPrototypeOf(tbl1), tableProto);
}, "'WebAssembly.Table' instance objects");

test(() => {
    const lengthDesc = Object.getOwnPropertyDescriptor(tableProto, 'length');
    assert_equals(typeof lengthDesc.get, "function");
    assert_equals(lengthDesc.set, undefined);
    assert_equals(lengthDesc.enumerable, true);
    assert_equals(lengthDesc.configurable, true);
}, "'WebAssembly.Table.prototype.length' accessor data property");

test(() => {
    const lengthDesc = Object.getOwnPropertyDescriptor(tableProto, 'length');
    const lengthGetter = lengthDesc.get;
    assert_equals(lengthGetter.length, 0);
    assertThrows(() => lengthGetter.call(), TypeError);
    assertThrows(() => lengthGetter.call({}), TypeError);
    assert_equals(typeof lengthGetter.call(tbl1), "number");
    assert_equals(lengthGetter.call(tbl1), 2);
}, "'WebAssembly.Table.prototype.length' getter");

test(() => {
    const getDesc = Object.getOwnPropertyDescriptor(tableProto, 'get');
    assert_equals(typeof getDesc.value, "function");
    assert_equals(getDesc.enumerable, true);
    assert_equals(getDesc.configurable, true);
}, "'WebAssembly.Table.prototype.get' data property");

test(() => {
    const getDesc = Object.getOwnPropertyDescriptor(tableProto, 'get');
    const get = getDesc.value;
    assert_equals(get.length, 1);
    assertThrows(() => get.call(), TypeError);
    assertThrows(() => get.call({}), TypeError);
    assert_equals(get.call(tbl1, 0), null);
    assert_equals(get.call(tbl1, 1), null);
    assert_equals(get.call(tbl1, 1.5), null);
    assertThrows(() => get.call(tbl1, 2), RangeError);
    assertThrows(() => get.call(tbl1, 2.5), RangeError);
    assertThrows(() => get.call(tbl1, -1), TypeError);
    assertThrows(() => get.call(tbl1, Math.pow(2,33)), TypeError);
    assertThrows(() => get.call(tbl1, {valueOf() { throw new Error("hi") }}), Error);
}, "'WebAssembly.Table.prototype.get' method");

test(() => {
    const setDesc = Object.getOwnPropertyDescriptor(tableProto, 'set');
    assert_equals(typeof setDesc.value, "function");
    assert_equals(setDesc.enumerable, true);
    assert_equals(setDesc.configurable, true);
}, "'WebAssembly.Table.prototype.set' data property");

test(() => {
    const setDesc = Object.getOwnPropertyDescriptor(tableProto, 'set');
    const set = setDesc.value;
    assert_equals(set.length, 2);
    assertThrows(() => set.call(), TypeError);
    assertThrows(() => set.call({}), TypeError);
    assertThrows(() => set.call(tbl1, 0), TypeError);
    assertThrows(() => set.call(tbl1, 2, null), RangeError);
    assertThrows(() => set.call(tbl1, -1, null), TypeError);
    assertThrows(() => set.call(tbl1, Math.pow(2,33), null), TypeError);
    assertThrows(() => set.call(tbl1, 0, undefined), TypeError);
    assertThrows(() => set.call(tbl1, 0, {}), TypeError);
    assertThrows(() => set.call(tbl1, 0, function() {}), TypeError);
    assertThrows(() => set.call(tbl1, 0, Math.sin), TypeError);
    assertThrows(() => set.call(tbl1, {valueOf() { throw Error("hai") }}, null), Error);
    assert_equals(set.call(tbl1, 0, null), undefined);
    assert_equals(set.call(tbl1, 1, null), undefined);
}, "'WebAssembly.Table.prototype.set' method");

test(() => {
    const tblGrowDesc = Object.getOwnPropertyDescriptor(tableProto, 'grow');
    assert_equals(typeof tblGrowDesc.value, "function");
    assert_equals(tblGrowDesc.enumerable, true);
    assert_equals(tblGrowDesc.configurable, true);
}, "'WebAssembly.Table.prototype.grow' data property");

test(() => {
    const tblGrowDesc = Object.getOwnPropertyDescriptor(tableProto, 'grow');
    const tblGrow = tblGrowDesc.value;
    assert_equals(tblGrow.length, 1);
    assertThrows(() => tblGrow.call(), TypeError);
    assertThrows(() => tblGrow.call({}), TypeError);
    assertThrows(() => tblGrow.call(tbl1, -1), TypeError);
    assertThrows(() => tblGrow.call(tbl1, Math.pow(2,32)), TypeError);
    var tbl = new Table({element:"anyfunc", initial:1, maximum:2});
    assert_equals(tbl.length, 1);
    assert_equals(tbl.grow(0), 1);
    assert_equals(tbl.length, 1);
    assert_equals(tbl.grow(1), 1);
    assert_equals(tbl.length, 2);
    assertThrows(() => tbl.grow(1), Error);
}, "'WebAssembly.Table.prototype.grow' method");

test(() => {
    const globalDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Global');
    assert_equals(typeof globalDesc.value, "function");
    assert_equals(globalDesc.writable, true);
    assert_equals(globalDesc.enumerable, false);
    assert_equals(globalDesc.configurable, true);
    Global = WebAssembly.Global;
}, "'WebAssembly.Global' data property");

test(() => {
    const globalDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'Global');
    assert_equals(Global, globalDesc.value);
    assert_equals(Global.length, 1);
    assert_equals(Global.name, "Global");
    assertThrows(() => Global(), TypeError);
    assertThrows(() => new Global(1), TypeError);
    assertThrows(() => new Global({}), TypeError);
    assertThrows(() => new Global({value: 'foo'}), TypeError);
    assertThrows(() => new Global({value: 'i64'}, 0), TypeError);
    assert_equals(new Global({value:'i32'}) instanceof Global, true);
    assert_equals(new Global({value:'i64'}) instanceof Global, true);
    assert_equals(new Global({value:'f32'}) instanceof Global, true);
    assert_equals(new Global({value:'f64'}) instanceof Global, true);
    assert_equals(new Global({value:'i32', mutable: false}) instanceof Global, true);
    assert_equals(new Global({value:'i64', mutable: false}) instanceof Global, true);
    assert_equals(new Global({value:'f64', mutable: false}) instanceof Global, true);
    assert_equals(new Global({value:'f64', mutable: false}) instanceof Global, true);
    assert_equals(new Global({value:'i32', mutable: true}) instanceof Global, true);
    assert_equals(new Global({value:'i64', mutable: true}) instanceof Global, true);
    assert_equals(new Global({value:'f64', mutable: true}) instanceof Global, true);
    assert_equals(new Global({value:'f64', mutable: true}) instanceof Global, true);
    assert_equals(new Global({value:'i32'}, 0x132) instanceof Global, true);
    assert_equals(new Global({value:'f32'}, 0xf32) instanceof Global, true);
    assert_equals(new Global({value:'f64'}, 0xf64) instanceof Global, true);
    assert_equals(new Global({value:'i32', mutable: false}, 0x132) instanceof Global, true);
    assert_equals(new Global({value:'f32', mutable: false}, 0xf32) instanceof Global, true);
    assert_equals(new Global({value:'f64', mutable: false}, 0xf64) instanceof Global, true);
    assert_equals(new Global({value:'i32', mutable: true}, 0x132) instanceof Global, true);
    assert_equals(new Global({value:'f32', mutable: true}, 0xf32) instanceof Global, true);
    assert_equals(new Global({value:'f64', mutable: true}, 0xf64) instanceof Global, true);
}, "'WebAssembly.Global' constructor function");

test(() => {
    const globalProtoDesc = Object.getOwnPropertyDescriptor(Global, 'prototype');
    assert_equals(typeof globalProtoDesc.value, "object");
    assert_equals(globalProtoDesc.writable, false);
    assert_equals(globalProtoDesc.enumerable, false);
    assert_equals(globalProtoDesc.configurable, false);
}, "'WebAssembly.Global.prototype' data property");

test(() => {
    const globalProtoDesc = Object.getOwnPropertyDescriptor(Global, 'prototype');
    globalProto = Global.prototype;
    assert_equals(globalProto, globalProtoDesc.value);
    assert_equals(String(globalProto), "[object WebAssembly.Global]");
    assert_equals(Object.getPrototypeOf(globalProto), Object.prototype);
}, "'WebAssembly.Global.prototype' object");

test(() => {
    globalI32 = new Global({value: 'i32'}, 0x132);
    globalF32 = new Global({value: 'f32'}, 0xf32);
    globalF64 = new Global({value: 'f64'}, 0xf64);
    globalI32Mut = new Global({value: 'i32', mutable: true}, 0x132);
    globalF32Mut = new Global({value: 'f32', mutable: true}, 0xf32);
    globalF64Mut = new Global({value: 'f64', mutable: true}, 0xf64);
    assert_equals(typeof globalI32, "object");
    assert_equals(String(globalI32), "[object WebAssembly.Global]");
    assert_equals(Object.getPrototypeOf(globalI32), globalProto);
}, "'WebAssembly.Global' instance objects");

test(() => {
    let builder = new WasmModuleBuilder();
    builder.addGlobal(kWasmI32).exportAs('i32');
    builder.addGlobal(kWasmI64).exportAs('i64');
    builder.addGlobal(kWasmF32).exportAs('f32');
    builder.addGlobal(kWasmF64).exportAs('f64');
    builder.addGlobal(kWasmI32, true).exportAs('i32mut');
    builder.addGlobal(kWasmI64, true).exportAs('i64mut');
    builder.addGlobal(kWasmF32, true).exportAs('f32mut');
    builder.addGlobal(kWasmF64, true).exportAs('f64mut');
    let module = new WebAssembly.Module(builder.toBuffer());
    let instance = new WebAssembly.Instance(module);

    assert_true(instance.exports.i32 instanceof WebAssembly.Global);
    assert_true(instance.exports.i64 instanceof WebAssembly.Global);
    assert_true(instance.exports.f32 instanceof WebAssembly.Global);
    assert_true(instance.exports.f64 instanceof WebAssembly.Global);
    assert_true(instance.exports.i32mut instanceof WebAssembly.Global);
    assert_true(instance.exports.i64mut instanceof WebAssembly.Global);
    assert_true(instance.exports.f32mut instanceof WebAssembly.Global);
    assert_true(instance.exports.f64mut instanceof WebAssembly.Global);

    // Can't set value of immutable globals.
    assertThrows(() => instance.exports.i32.value = 0, TypeError);
    assertThrows(() => instance.exports.i64.value = 0, TypeError);
    assertThrows(() => instance.exports.f32.value = 0, TypeError);
    assertThrows(() => instance.exports.f64.value = 0, TypeError);

    instance.exports.i32mut.value = 13579;
    instance.exports.f32mut.value = 24680;
    instance.exports.f64mut.value = 124816;

    globalI64 = instance.exports.i64;
    globalI64Mut = instance.exports.i64mut;
}, "Export 'WebAssembly.Global'");

test(() => {
    const lengthDesc = Object.getOwnPropertyDescriptor(globalProto, 'value');
    assert_equals(typeof lengthDesc.get, "function");
    assert_equals(typeof lengthDesc.set, "function");
    assert_equals(lengthDesc.enumerable, true);
    assert_equals(lengthDesc.configurable, true);
}, "'WebAssembly.Global.prototype.value' accessor data property");

test(() => {
    const valueDesc = Object.getOwnPropertyDescriptor(globalProto, 'value');
    const valueGetter = valueDesc.get;
    assert_equals(valueGetter.length, 0);
    assertThrows(() => valueGetter.call(), TypeError);
    assertThrows(() => valueGetter.call({}), TypeError);

    assert_equals(typeof valueGetter.call(globalI32), "number");
    assertThrows(() => valueGetter.call(globalI64), TypeError);
    assert_equals(typeof valueGetter.call(globalF32), "number");
    assert_equals(typeof valueGetter.call(globalF64), "number");
    assert_equals(typeof valueGetter.call(globalI32Mut), "number");
    assertThrows(() => valueGetter.call(globalI64Mut), TypeError);
    assert_equals(typeof valueGetter.call(globalF32Mut), "number");
    assert_equals(typeof valueGetter.call(globalF64Mut), "number");

    assert_equals(valueGetter.call(globalI32), 0x132);
    assert_equals(valueGetter.call(globalF32), 0xf32);
    assert_equals(valueGetter.call(globalF64), 0xf64);
    assert_equals(valueGetter.call(globalI32Mut), 0x132);
    assert_equals(valueGetter.call(globalF32Mut), 0xf32);
    assert_equals(valueGetter.call(globalF64Mut), 0xf64);
}, "'WebAssembly.Global.prototype.value' getter");

test(() => {
    const valueDesc = Object.getOwnPropertyDescriptor(globalProto, 'value');
    const valueSetter = valueDesc.set;
    assert_equals(valueSetter.length, 1);
    assertThrows(() => valueSetter.call(), TypeError);
    assertThrows(() => valueSetter.call({}), TypeError);

    assertThrows(() => valueSetter.call(globalI32, 1234), TypeError);
    assertThrows(() => valueSetter.call(globalI64, 1234), TypeError);
    assertThrows(() => valueSetter.call(globalF32, 1234), TypeError);
    assertThrows(() => valueSetter.call(globalF64, 1234), TypeError);

    valueSetter.call(globalI32Mut, 1234);
    assertThrows(() => valueSetter.call(globalI64Mut, 1234), TypeError);
    valueSetter.call(globalF32Mut, 5678);
    valueSetter.call(globalF64Mut, 9012);

    assert_equals(globalI32Mut.value, 1234);
    assert_equals(globalF32Mut.value, 5678);
    assert_equals(globalF64Mut.value, 9012);
}, "'WebAssembly.Global.prototype.value' setter");

test(() => {
    const valueOfDesc = Object.getOwnPropertyDescriptor(globalProto, 'valueOf');
    assert_equals(typeof valueOfDesc.value, "function");
    assert_equals(valueOfDesc.enumerable, true);
    assert_equals(valueOfDesc.configurable, true);
}, "'WebAssembly.Global.prototype.valueOf' data property");

test(() => {
    const valueOfDesc = Object.getOwnPropertyDescriptor(globalProto, 'valueOf');
    const valueOf = valueOfDesc.value;
    assert_equals(valueOf.length, 0);
    assertThrows(() => valueOf.call(), TypeError);
    assertThrows(() => valueOf.call({}), TypeError);

    assert_equals(valueOf.call(globalI32), 0x132);
    assertThrows(() => valueOf.call(globalI64), TypeError);
    assert_equals(valueOf.call(globalF32), 0xf32);
    assert_equals(valueOf.call(globalF64), 0xf64);
    assert_equals(valueOf.call(globalI32Mut), 1234);
    assertThrows(() => valueOf.call(globalI64Mut), TypeError);
    assert_equals(valueOf.call(globalF32Mut), 5678);
    assert_equals(valueOf.call(globalF64Mut), 9012);
}, "'WebAssembly.Global.prototype.valueOf' method");

test(() => {
    assert_equals(new Global({value: 'i32'}).value, 0);
    assert_equals(new Global({value: 'f32'}).value, 0);
    assert_equals(new Global({value: 'f64'}).value, 0);
}, "'WebAssembly.Global' default value is 0");

test(() => {
    let builder = new WasmModuleBuilder();
    builder.addImportedGlobal('', 'i32', kWasmI32);
    builder.addImportedGlobal('', 'i64', kWasmI64);
    builder.addImportedGlobal('', 'f32', kWasmF32);
    builder.addImportedGlobal('', 'f64', kWasmF64);
    builder.addImportedGlobal('', 'i32mut', kWasmI32, true);
    builder.addImportedGlobal('', 'i64mut', kWasmI64, true);
    builder.addImportedGlobal('', 'f32mut', kWasmF32, true);
    builder.addImportedGlobal('', 'f64mut', kWasmF64, true);
    let module = new WebAssembly.Module(builder.toBuffer());
    let instance = new WebAssembly.Instance(module, {
        '': {
          i32: globalI32,
          i64: globalI64,
          f32: globalF32,
          f64: globalF64,
          i32mut: globalI32Mut,
          i64mut: globalI64Mut,
          f32mut: globalF32Mut,
          f64mut: globalF64Mut
        }
    });
}, "Import 'WebAssembly.Global'");

test(() => {
    let assertInstanceError = (type, mutable, imports) => {
      assertThrows(() => {
        let builder = new WasmModuleBuilder();
        builder.addImportedGlobal('', 'g', type, mutable);
        let module = new WebAssembly.Module(builder.toBuffer());
        let instance = new WebAssembly.Instance(module, imports);
      }, LinkError);
    };

    const immutable = false, mutable = true;

    // Type mismatch.
    assertInstanceError(kWasmI32, immutable, {'': {g: globalI64}});
    assertInstanceError(kWasmI32, immutable, {'': {g: globalF32}});
    assertInstanceError(kWasmI32, immutable, {'': {g: globalF64}});
    assertInstanceError(kWasmI64, immutable, {'': {g: globalI32}});
    assertInstanceError(kWasmI64, immutable, {'': {g: globalF32}});
    assertInstanceError(kWasmI64, immutable, {'': {g: globalF64}});
    assertInstanceError(kWasmF32, immutable, {'': {g: globalI32}});
    assertInstanceError(kWasmF32, immutable, {'': {g: globalI64}});
    assertInstanceError(kWasmF32, immutable, {'': {g: globalF64}});
    assertInstanceError(kWasmF64, immutable, {'': {g: globalI32}});
    assertInstanceError(kWasmF64, immutable, {'': {g: globalI64}});
    assertInstanceError(kWasmF64, immutable, {'': {g: globalF32}});

    assertInstanceError(kWasmI32, mutable, {'': {g: globalI64Mut}});
    assertInstanceError(kWasmI32, mutable, {'': {g: globalF32Mut}});
    assertInstanceError(kWasmI32, mutable, {'': {g: globalF64Mut}});
    assertInstanceError(kWasmI64, mutable, {'': {g: globalI32Mut}});
    assertInstanceError(kWasmI64, mutable, {'': {g: globalF32Mut}});
    assertInstanceError(kWasmI64, mutable, {'': {g: globalF64Mut}});
    assertInstanceError(kWasmF32, mutable, {'': {g: globalI32Mut}});
    assertInstanceError(kWasmF32, mutable, {'': {g: globalI64Mut}});
    assertInstanceError(kWasmF32, mutable, {'': {g: globalF64Mut}});
    assertInstanceError(kWasmF64, mutable, {'': {g: globalI32Mut}});
    assertInstanceError(kWasmF64, mutable, {'': {g: globalI64Mut}});
    assertInstanceError(kWasmF64, mutable, {'': {g: globalF32Mut}});

    // Mutable mismatch.
    assertInstanceError(kWasmI32, immutable, {'': {g: globalI32Mut}});
    assertInstanceError(kWasmI64, immutable, {'': {g: globalI64Mut}});
    assertInstanceError(kWasmF32, immutable, {'': {g: globalF32Mut}});
    assertInstanceError(kWasmF64, immutable, {'': {g: globalF64Mut}});

    assertInstanceError(kWasmI32, mutable, {'': {g: globalI32}});
    assertInstanceError(kWasmI64, mutable, {'': {g: globalI64}});
    assertInstanceError(kWasmF32, mutable, {'': {g: globalF32}});
    assertInstanceError(kWasmF64, mutable, {'': {g: globalF64}});

    // Can't import Number as mutable.
    assertInstanceError(kWasmI32, mutable, {'': {g: 1}});
    assertInstanceError(kWasmI64, mutable, {'': {g: 1}});
    assertInstanceError(kWasmF32, mutable, {'': {g: 1}});
    assertInstanceError(kWasmF64, mutable, {'': {g: 1}});
}, "Import 'WebAssembly.Global' type mismatch");

test(() => {
    assertThrows(() => WebAssembly.validate(), TypeError);
    assertThrows(() => WebAssembly.validate("hi"), TypeError);
    assert_true(WebAssembly.validate(emptyModuleBinary));
    assert_true(WebAssembly.validate(complexImportingModuleBinary));
    assert_false(WebAssembly.validate(moduleBinaryImporting2Memories));
    assert_false(WebAssembly.validate(moduleBinaryWithMemSectionAndMemImport));
}, "'WebAssembly.validate' method");

test(() => {
    const compileDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'compile');
    assert_equals(typeof compileDesc.value, "function");
    assert_equals(compileDesc.writable, true);
    assert_equals(compileDesc.enumerable, true);
    assert_equals(compileDesc.configurable, true);
}, "'WebAssembly.compile' data property");

test(() => {
    const compile = WebAssembly.compile;
    const compileDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'compile');

    assert_equals(compile, compileDesc.value);
    assert_equals(compile.length, 1);
    assert_equals(compile.name, "compile");
}, "'WebAssembly.compile' function");

var num_tests = 1;
function assertCompileError(args, err) {
    promise_test(() => {
        return WebAssembly.compile(...args)
        .then(_ => {
            throw null;
        })
        .catch(error => {
            assert_equals(error instanceof err, true);
            return Promise.resolve()
        });
    }, `assertCompileError ${num_tests++}`);
}

assertCompileError([], TypeError);
assertCompileError([undefined], TypeError);
assertCompileError([1], TypeError);
assertCompileError([{}], TypeError);
assertCompileError([new Uint8Array()], CompileError);
assertCompileError([new ArrayBuffer()], CompileError);
assertCompileError([kC0DEFEFE], CompileError);

num_tests = 1;
function assertCompileSuccess(bytes) {
    promise_test(() => {
        return WebAssembly.compile(bytes)
        .then(module => {
            assert_equals(module instanceof Module, true);
        });
    }, `assertCompileSuccess ${num_tests++}`);
}

assertCompileSuccess(emptyModuleBinary);
assertCompileSuccess(new Uint8Array(emptyModuleBinary));

test(() => {
    const instantiateDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'instantiate');
    assert_equals(typeof instantiateDesc.value, "function");
    assert_equals(instantiateDesc.writable, true);
    assert_equals(instantiateDesc.enumerable, true);
    assert_equals(instantiateDesc.configurable, true);
}, "'WebAssembly.instantiate' data property");

test(() => {
    const instantiateDesc = Object.getOwnPropertyDescriptor(WebAssembly, 'instantiate');
    const instantiate = WebAssembly.instantiate;
    assert_equals(instantiate, instantiateDesc.value);
    assert_equals(instantiate.length, 1);
    assert_equals(instantiate.name, "instantiate");
    function assertInstantiateError(args, err) {
        promise_test(() => {
            return instantiate(...args)
                .then(m => {
                    throw null;
                })
                .catch(error => {
                    assert_equals(error instanceof err, true);
                })
        }, 'unexpected success in assertInstantiateError');
    }
    var scratch_memory = new WebAssembly.Memory({initial:1});
    var scratch_table = new WebAssembly.Table({element:"anyfunc", initial:1, maximum:1});
    assertInstantiateError([], TypeError);
    assertInstantiateError([undefined], TypeError);
    assertInstantiateError([1], TypeError);
    assertInstantiateError([{}], TypeError);
    assertInstantiateError([new Uint8Array()], CompileError);
    assertInstantiateError([new ArrayBuffer()], CompileError);
    assertInstantiateError([kC0DEFEFE], CompileError);
    assertInstantiateError([importingModule], TypeError);
    assertInstantiateError([importingModule, null], TypeError);
    assertInstantiateError([importingModuleBinary, null], TypeError);
    assertInstantiateError([emptyModule, null], TypeError);
    assertInstantiateError([importingModuleBinary, null], TypeError);
    assertInstantiateError([importingModuleBinary, undefined], TypeError);
    assertInstantiateError([importingModuleBinary, {}], TypeError);
    assertInstantiateError([importingModuleBinary, {"":{g:()=>{}}}], LinkError);
    assertInstantiateError([importingModuleBinary, {t:{f:()=>{}}}], TypeError);
    assertInstantiateError([complexImportingModuleBinary, null], TypeError);
    assertInstantiateError([complexImportingModuleBinary, undefined], TypeError);
    assertInstantiateError([complexImportingModuleBinary, {}], TypeError);
    assertInstantiateError([complexImportingModuleBinary, {"c": {"d": scratch_memory}}], TypeError);

    function assertInstantiateSuccess(module, imports) {
        promise_test(()=> {
            return instantiate(module, imports)
                .then(result => {
                    if (module instanceof Module) {
                        assert_equals(result instanceof Instance, true);
                    } else {
                        assert_equals(result.module instanceof Module, true);
                        assert_equals(result.instance instanceof Instance, true);
                        var desc = Object.getOwnPropertyDescriptor(result, 'module');
                        assert_equals(desc.writable, true);
                        assert_equals(desc.enumerable, true);
                        assert_equals(desc.configurable, true);
                        desc = Object.getOwnPropertyDescriptor(result, 'instance');
                        assert_equals(desc.writable, true);
                        assert_equals(desc.enumerable, true);
                        assert_equals(desc.configurable, true);
                    }
                })}, 'unexpected failure in assertInstantiateSuccess');
    }
    assertInstantiateSuccess(emptyModule);
    assertInstantiateSuccess(emptyModuleBinary);
    assertInstantiateSuccess(new Uint8Array(emptyModuleBinary));
    assertInstantiateSuccess(importingModule, {"":{f:()=>{}}});
    assertInstantiateSuccess(importingModuleBinary, {"":{f:()=>{}}});
    assertInstantiateSuccess(new Uint8Array(importingModuleBinary), {"":{f:()=>{}}});
    assertInstantiateSuccess(complexImportingModuleBinary, {
        a:{b:()=>{}},
        c:{d:scratch_memory},
        e:{f:scratch_table},
        g:{'⚡':1}});
}, "'WebAssembly.instantiate' function");

const complexReExportingModuleBinary = (() => {
    let builder = new WasmModuleBuilder();

    let fIndex = builder.addImport('a', 'f', kSig_i_i);
    let gIndex = builder.addImport('a', 'g', kSig_i_v);
    builder.addImportedMemory('c', 'd');
    builder.addImportedTable('e', 'f');

    builder.addExport('x', fIndex)
    builder.addExport('y', gIndex)
    builder.addExportOfKind('z', kExternalMemory, 0)
    builder.addExportOfKind('w', kExternalTable, 0)

    return builder.toBuffer();
})();

const complexTableReExportingModuleBinary = (() => {
    let builder = new WasmModuleBuilder();

    builder.addImport('a', '_', kSig_v_v);
    let gIndex = builder.addImport('a', 'g', kSig_i_v);
    let fIndex = builder.addImport('a', 'f', kSig_i_i);
    let hIndex = builder
        .addFunction('h', kSig_i_v)
        .addBody([
            kExprI32Const,
            46,
            kExprEnd
        ]).index;

    builder.setFunctionTableLength(3);
    builder.appendToTable([fIndex, gIndex, hIndex]);
    builder.addExportOfKind('tab', kExternalTable, 0);

    return builder.toBuffer();
})();


test(() => {
  let module = new WebAssembly.Module(complexReExportingModuleBinary);
  let memory = new WebAssembly.Memory({initial: 0});
  let table = new WebAssembly.Table({initial: 0, element: 'anyfunc'});
  let imports = {
    a: { f(x) { return x+1; }, g: exportingInstance.exports.f },
    c: { d: memory },
    e: { f: table },
  };
  let instance = reExportingInstance =
      new WebAssembly.Instance(module, imports);

  assert_equals(instance.exports.x.name, "0");
  assert_false(instance.exports.x === imports.a.f);

  // Previously exported Wasm functions are re-exported with the same value
  assert_equals(instance.exports.y, exportingInstance.exports.f);
  assert_equals(instance.exports.y.name, "0");

  // Re-exported Memory and Table objects have the same value
  assert_equals(instance.exports.z, memory);
  assert_equals(instance.exports.w, table);

  // Importing the same JS function object results in a distinct exported
  // function object, whereas previously exported Wasm functions are
  // re-exported with the same identity.
  let instance2 = new WebAssembly.Instance(module, imports);
  assert_false(instance.exports.x === instance2.exports.x);
  assert_equals(instance.exports.y, instance2.exports.y);
}, "Exported values have cached JS objects");

test(() => {
  let module = new WebAssembly.Module(complexTableReExportingModuleBinary);
  let instance = new WebAssembly.Instance(module,
      { a: { _() { }, f: reExportingInstance.exports.x,
             g: exportingInstance.exports.f } });
  let table = instance.exports.tab;

  // The functions that were put in come right back out
  assert_equals(table.get(0), reExportingInstance.exports.x);
  assert_equals(table.get(1), exportingInstance.exports.f);

  // The original function indices are reflected in the name
  assert_equals(table.get(0).name, "0");
  assert_equals(table.get(1).name, "0");
  assert_equals(table.get(2).name, "3");

  // All of the functions work
  assert_equals(table.get(0)(5), 6);
  assert_equals(table.get(1)(), 42);
  assert_equals(table.get(2)(), 46);
}, "Tables export cached");

test(() => {
  let module = new WebAssembly.Module(exportingModuleIdentityFn );
  let instance = new WebAssembly.Instance(module);

  let value = 2 ** 31;
  let output = instance.exports.id(value);
  assert_equals(output, - (2 ** 31));
}, "WebAssembly integers are converted to JavaScript as if by ToInt32");

})();
