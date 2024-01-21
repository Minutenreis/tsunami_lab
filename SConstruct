##
# @author Alexander Breuer (alex.breuer AT uni-jena.de)
#
# @section DESCRIPTION
# Entry-point for builds.
##
import SCons
import distro
import nvcc
import os

print( '####################################' )
print( '### Tsunami Lab                  ###' )
print( '###                              ###' )
print( '### https://scalable.uni-jena.de ###' )
print( '####################################' )
print()
print('running build script')

# configuration
vars = Variables()

vars.AddVariables(
  EnumVariable( 'mode',
                'compile modes, option \'san\' enables address and undefined behavior sanitizers',
                'release',
                allowed_values=('release', 'debug', 'release+san', 'debug+san' )
              )
)

# exit in the case of unknown variables
if vars.UnknownVariables():
  print( "build configuration corrupted, don't know what to do with: " + str(vars.UnknownVariables().keys()) )
  exit(1)
  
# create environment
env = Environment( variables = vars )


nvcc.generate(env)

# check for libs
conf = Configure(env)
if not conf.CheckLibWithHeader('netcdf','netcdf.h','c++'):
  print('Did not find netcdf.h, exiting!')
  exit(1)

# set compiler
cxxCompiler = ARGUMENTS.get('comp', "g++")

# workaround to find the right g++ version on Ara
if 'centos' == distro.id():
  if cxxCompiler == 'g++':
    print('running on Ara, using gcc-11.2.0')
    env.Replace(CXX="/cluster/spack/opt/spack/linux-centos7-broadwell/gcc-10.2.0/gcc-11.2.0-c27urtyjryzoyyqfms5m3ewi6vrtvt44/bin/g++")
  else:    
    print('running on Ara, using icpc-19.1.2.254')
    env.Replace(CXX="/cluster/intel/parallel_studio_xe_2020.2.108/compilers_and_libraries_2020/linux/bin/intel64/icpc")
else:
  if cxxCompiler == 'g++':
    pass
  else:
    env.Replace(CXX="/opt/intel/oneapi/compiler/2023.2.2/linux/bin/intel64/icpc")

# generate help message
Help( vars.GenerateHelpText( env ) )

# add default flags
env.Append( CXXFLAGS = [ '-std=c++17',
                           '-Wall',
                           '-Wextra',
                           '-lcuda',
                           '-g',
                           '-march=native',
                           '-mtune=native',
                           '-Werror',])
if( 'g++' == cxxCompiler ):
  env.Append( CXXFLAGS = [ '-Wpedantic',
                           '-fopenmp'] )
  env.Append( LINKFLAGS = [ '-fopenmp', '-lcudart'] )
else:
  env.Append( CXXFLAGS = ['-diag-disable=10441',
                           '-wd823',
                           '-qopenmp'] )
  env.Append( LINKFLAGS = [ '-qopenmp', '-lcudart'] )
  optReport = ARGUMENTS.get('optReport', "false")
  if optReport!='false': 
    env.Append( CXXFLAGS = ['-qopt-report=5'])
  
  
# set optimization mode
if 'debug' in env['mode']:
  env.Append( CXXFLAGS = [ '-g',
                           '-O0' ] )
  print( 'using optimization flag: -O0 -g' )
else:
  cxxOptimization = ARGUMENTS.get('cxxO', "-O3")
  env.Append( CXXFLAGS = [ cxxOptimization ] )
  print( 'using optimization flag: ' + cxxOptimization )

# add sanitizers
if 'san' in  env['mode']:
  env.Append( CXXFLAGS =  [ '-g',
                            '-fsanitize=float-divide-by-zero',
                            '-fsanitize=bounds',
                            '-fsanitize=address',
                            '-fsanitize=undefined',
                            '-fno-omit-frame-pointer' ] )
  env.Append( LINKFLAGS = [ '-g',
                            '-fsanitize=address',
                            '-fsanitize=undefined' ] )

# add Catch2
env.Append( CXXFLAGS = [ '-isystem', 'submodules/Catch2/single_include' ] )
env.Append( CXXFLAGS = [ '-isystem', 'submodules/json/single_include'] )

# add CUDA
env.Append( LIBPATH = ["/usr/local/cuda/lib64"] ) 
env.Append(LIBS=['cudart'])

# get source files
VariantDir( variant_dir = 'build/src',
            src_dir     = 'src' )

env.sources = []
env.tests = []

# CUDA source files
cuda_sources = env.Glob('src/*.cu')
cuda_objects = [env.Object(target='build/src/cuda_objects/%s' % os.path.basename(src), source=src) for src in cuda_sources]
env.sources += cuda_objects

Export('env')
SConscript( 'build/src/SConscript' )
Import('env')

env.Program( target = 'build/tsunami_lab',
             source = env.sources + env.standalone )

env.Program( target = 'build/tests',
             source = env.sources + env.tests )