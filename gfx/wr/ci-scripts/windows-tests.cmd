:: This Source Code Form is subject to the terms of the Mozilla Public
:: License, v. 2.0. If a copy of the MPL was not distributed with this
:: file, You can obtain one at http://mozilla.org/MPL/2.0/. */

:: This must be run from the root webrender directory!
:: Users may set the CARGOFLAGS environment variable to pass
:: additional flags to cargo if desired.

pushd webrender_api
cargo test
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
popd

pushd webrender
cargo test --features gl
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
cargo check
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
popd

pushd wrench
:: cargo check --features vulkan
:: if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
cargo check --features dx12
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
:: cargo test --features gl
:: if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
cargo run --release --features gl -- --angle reftest
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
popd

pushd examples
cargo check --features gl
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
cargo check --features dx12
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
:: cargo check --features vulkan
:: if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
popd

pushd direct-composition
cargo check
if %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
popd
