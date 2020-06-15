<template>
  <div id="app">
    <Navigation/>
    <MainContainer/>
  </div>
</template>

<script>
import Navigation from './components/Navigation'
import MainContainer from './components/MainContainer'

export default {
  name: 'App',
  components: {MainContainer, Navigation}
}
</script>

<style>
  #app {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    margin-top: 0;
    padding-top: 0;
  }
  body {
    margin: 0;
  }
</style>
